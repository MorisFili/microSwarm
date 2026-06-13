package derivative.code.microswarm.entity

import derivative.code.microswarm.INV_COUNT
import derivative.code.microswarm.INV_DETECTION_RADIUS
import derivative.code.microswarm.INV_TARGET_RADIUS
import derivative.code.microswarm.Main
import derivative.code.microswarm.Simulation
import derivative.code.microswarm.Simulation.Companion.rng
import derivative.code.microswarm.TABLE_SIZE
import derivative.code.microswarm.cosTable
import derivative.code.microswarm.managePopHueCounter
import derivative.code.microswarm.network.Network
import derivative.code.microswarm.sinTable
import javafx.scene.paint.Color
import java.util.Arrays
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sign


open class Agent(
    // Coordinates
    x: Float,
    y: Float,

    // Attributes
    id: Int,
    hue: Color,
    isEnabled: Boolean = true,
    private val network: Network
) : Entity(id, x, y, hue, isEnabled) {

    // Field variables
    val detectionRadius = 40f // Max possible radius = 50 in the current proximity scan logic due to cell size
    val targetRadius = detectionRadius / 2f
    val actionRadius = targetRadius / 10f
    val PRESENCE_CAP = 50
    val GROUP_SIZE = PRESENCE_CAP / 3f
    val agentsInProximity = arrayOfNulls<Agent>(PRESENCE_CAP)
    val entitiesInProximity = arrayOfNulls<Entity>(5)
    var withinActionRadius = false
    var TARGET: Agent? = null
    val targetedBy = arrayOfNulls<Agent>(20)
    var foodField: Food? = null
    var foodFieldDistance = 1f
    var MATE_CONDITION = false
    var INCUBATING = false
    var INCUBATION_TIMER = 0f
    var INCUBATION_MATERIAL: GeneticMaterial? = null
    val preState = FloatArray(network.inputStates)
    val postState = FloatArray(network.inputStates)
    val preMotorInputs = FloatArray(network.motionInputs)
    val postMotorInputs = FloatArray(network.motionInputs)
    var facingX = 0f
    var facingY = 1f
    var explorationTargetX = 0f
    var explorationTargetY = 0f
    var metaData = FloatArray(15)

    var output = FloatArray(network.networkOutputs)
    var ENERGY = 100f
    val MAX_ENERGY = 100f

    // Social features
    var AGENTS_PRESENCE_COUNT = 0
    var ENTITIES_PRESENCE_COUNT = 0
    var REPUTATION = 1f
    var KILL_COUNT = 0f
    var targetCooldown = 0

    // Genetic traits
    var isMale = false // true = male
    var isApathic = false

    fun randomizeTraits() {
        isMale = rng.nextFloat() < 0.5
        isApathic = rng.nextFloat() < 0.1
    }

    init {
        randomizeTraits()
    }


    fun syncGenerateIntent() {
        if (targetCooldown > 0) {
            targetCooldown--
        }
        if (INCUBATION_TIMER > 0f) INCUBATION_TIMER--
        updateProximity()
        generateState(preState)
    }

    fun asyncFeedForward() {
        network.generateIntent(preState, output)
        generateIntentState(preMotorInputs)
        selectTarget()
    }

    fun syncPerformAction() {
        action(output[2], output[3], output[4])
        if (INCUBATING && INCUBATION_TIMER < 1f) {
            if (INCUBATION_MATERIAL != null) MATE_CONDITION = true
            INCUBATING = false
            INCUBATION_TIMER = 0f
            network.valence += 1f
        }
    }

    fun asyncActionEvaluation() {
        network.actionEvaluation()
        network.generateMovement(preMotorInputs, output)
    }

    fun syncMovement() {
        move(output[0], output[1], preMotorInputs[0])
    }

    fun asyncStateEvaluation() {
        generateState(postState)
        generateIntentState(postMotorInputs)
        network.stateEvaluation(preState, postState)
        network.movementEvaluation(preMotorInputs, postMotorInputs)
        network.weightAdjustment()
        ENERGY -= if (INCUBATING) 0.02f else 0.01f
        if (ENERGY <= 0f) starvation()
    }

    private fun selectTarget() {

        // ========= Targeted By Block =========
        var bestTargetedDistance = 1000f
        var closestTargetedBy: Agent? = null
        for (i in 0 until targetedBy.size) {
            val other = targetedBy[i] ?: continue
            val distance = cheapDistance(this, other) / targetRadius
            val urgency = other.KILL_COUNT * (1 - other.REPUTATION) * (1 - distance)
            if (urgency > 0.5f && distance < bestTargetedDistance && lineOfSightCheck(other)) {
                closestTargetedBy = other
                bestTargetedDistance = distance
            }
        }

        // ======== Possible targets ========
        var bestDistSq = 1000f
        var bestTarget: Agent? = null
        for (agent in agentsInProximity) {
            if (agent == null) continue
            if (!agent.enabled) continue
            if (lineOfSightCheck(agent)) {

                val dx = agent.x - x
                val dy = agent.y - y
                val targetDistSq = dx * dx + dy * dy

                // Pick closest to anchor
                if (targetDistSq < bestDistSq) {
                    bestDistSq = targetDistSq
                    bestTarget = agent
                }
            }
        }

        // Self defense preference
        if (targetCooldown < 1) {
            if (closestTargetedBy != null) {
                setTarget(closestTargetedBy)
            } else if (bestTarget != null) {
                setTarget(bestTarget)
            }
        }
    }

    private fun generateState(state: FloatArray) {


        // ========= Sense of Urgency Block =========

        var acuteScore = 0f
        for (i in 0 until targetedBy.size) {
            val other = targetedBy[i] ?: continue
            val distance = cheapDistance(this, other) / targetRadius
            acuteScore += other.KILL_COUNT * (1 - other.REPUTATION) * (1 - distance)
        }

        // ========= Target & Environment Sensing Block =========

        // Target clearing logic
        val targetRadiusSq = targetRadius * targetRadius
        if (targetCooldown > 0) clearTarget()
        val target = TARGET // Local pointer
        if (target != null && !target.enabled) clearTarget() // Clear valid but disabled target
        if (target != null) { // Old target that went out of range
            val dx = target.x - x
            val dy = target.y - y
            val targetDistSq = dx * dx + dy * dy
            if (targetDistSq > targetRadiusSq) clearTarget()
        }

        // Local agent analysis logic
        var familiarX = 0f
        var familiarY = 0f
        var familiarCount = 0f

        var unFamiliarX = 0f
        var unFamiliarY = 0f
        var unFamiliarCount = 0f

        var dangerX = 0f
        var dangerY = 0f
        var dangerCount = 0f
        var localDangerValue = 0f

        val maxRadiusSq = detectionRadius * detectionRadius
        for (agent in agentsInProximity) {
            if (agent == null) continue
            if (!agent.enabled) continue
            val dx = agent.x - x
            val dy = agent.y - y
            val distSq = dx * dx + dy * dy
            val normDistFactor = 1 - (distSq / maxRadiusSq) // 0 = far away, 1 = next to

            if (agent.hue == hue) { // Same hue
                familiarCount++
                familiarX += dx
                familiarY += dy
            } else { // Other hue count
                unFamiliarCount++
                unFamiliarX += dx
                unFamiliarY += dy
            }

            val dangerValue = agent.KILL_COUNT * (1 - agent.REPUTATION)
            if (dangerValue > 0.5) {
                dangerX += dx
                dangerY += dy
                dangerCount++
                localDangerValue += dangerValue * normDistFactor
            }

        }

        withinActionRadius = actionRadiusCheck(TARGET)

        // Local resource analysis logic
        var foodValue = 0f
        var closestFood: Food? = null
        var closestDistSqFood = 1000f
        for (entity in entitiesInProximity) { // Non-agent entity array
            if (entity is Food) {
                if (!entity.enabled) continue
                val dx = entity.x - x
                val dy = entity.y - y
                val distSq = dx * dx + dy * dy
                val prox = (1f - distSq / maxRadiusSq).coerceIn(0f, 1f)
                val ripeness = (entity.value / 25f).coerceIn(0f, 1f)
                val availability = (1f + entity.decaySpeed).coerceIn(0f, 1f)
                foodValue += prox * ripeness * availability

                if (distSq < closestDistSqFood) {
                    closestDistSqFood = distSq
                    closestFood = entity
                }
            }
        }
        // Crude gate that checks both for proximity and availability
        foodField = if (closestFood != null && closestDistSqFood < 100f) closestFood else null
        foodFieldDistance = if (closestFood != null) closestDistSqFood / (maxRadiusSq) else 1f

        // ========= Coordination Block =========

        // Familiar Group
        val famAvgDX =
            if (familiarCount > 0) familiarX * INV_COUNT[familiarCount.toInt()] * INV_DETECTION_RADIUS else 0f
        val famAvgDY =
            if (familiarCount > 0) familiarY * INV_COUNT[familiarCount.toInt()] * INV_DETECTION_RADIUS else 0f
        metaData[0] = famAvgDX * -facingY + famAvgDY * facingX
        metaData[1] = famAvgDX * facingX + famAvgDY * facingY
        val familiarGroupSize = familiarCount / GROUP_SIZE
        val familiarDistance = if (familiarCount > 0) cheapDistance(famAvgDX, famAvgDY) else 0f
        metaData[2] = familiarDistance

        // Unfamiliar Group
        val unFamAvgDX =
            if (unFamiliarCount > 0) unFamiliarX * INV_COUNT[unFamiliarCount.toInt()] * INV_DETECTION_RADIUS else 0f
        val unFamAvgDY =
            if (unFamiliarCount > 0) unFamiliarY * INV_COUNT[unFamiliarCount.toInt()] * INV_DETECTION_RADIUS else 0f
        metaData[3] = unFamAvgDX * -facingY + unFamAvgDY * facingX
        metaData[4] = unFamAvgDX * facingX + unFamAvgDY * facingY
        val unFamiliarGroupSize = unFamiliarCount / GROUP_SIZE
        val unFamiliarThreat = if (unFamiliarCount > 0) (1 - cheapDistance(unFamAvgDX, unFamAvgDY)) else 0f
        metaData[5] = unFamiliarThreat

        // Center of danger
        val dangerAvgDX = if (dangerCount > 0) dangerX * INV_COUNT[dangerCount.toInt()] * INV_DETECTION_RADIUS else 0f
        val dangerAvgDY = if (dangerCount > 0) dangerY * INV_COUNT[dangerCount.toInt()] * INV_DETECTION_RADIUS else 0f
        metaData[6] = dangerAvgDX * -facingY + dangerAvgDY * facingX
        metaData[7] = dangerAvgDX * facingX + dangerAvgDY * facingY
        val dangerCenterStrength = if (dangerCount > 0) localDangerValue * INV_COUNT[dangerCount.toInt()] else 0f
        val dangerCenterThreat = if (dangerCount > 0) (1f - cheapDistance(dangerAvgDX, dangerAvgDY)) else 0f
        metaData[8] = dangerCenterThreat

        // Target
        val targetDX = if (target != null) (target.x - x) * INV_TARGET_RADIUS else 0f
        val targetDY = if (target != null) (target.y - y) * INV_TARGET_RADIUS else 0f
        metaData[9] = targetDX * -facingY + targetDY * facingX
        metaData[10] = targetDX * facingX + targetDY * facingY
        val targetDistance = if (target != null) cheapDistance(this, target) * INV_TARGET_RADIUS else 0f
        metaData[11] = targetDistance
        val attractiveness = if (target != null) {
            val sexCompat = if (target.isMale != isMale) 1f else 0f
            val repQuality = target.REPUTATION.coerceIn(0f, 1f)
            val health = (target.ENERGY / target.MAX_ENERGY).coerceIn(0f, 1f)
            val available = if (!target.INCUBATING) 1f else 0f
            sexCompat * 0.5f + repQuality * 0.25f + health * 0.15f + available * 0.1f
        } else 0f


        val threat = if (target != null) {
            val dangerScore = (target.KILL_COUNT / (1f + target.KILL_COUNT)) *
                    (1f - target.REPUTATION).coerceIn(0f, 1f)
            val apathic = if (target.isApathic) 0.3f else 0f
            val huntingMe = if (targetedBy.any { it === target }) 1f else 0f
            (dangerScore * 0.5f + apathic * 0.2f + huntingMe * 0.3f).coerceIn(0f, 1f)
        } else 0f

        val targetEval = (attractiveness - threat).coerceIn(-1f, 1f)

        // Resources
        val resourceDX = if (closestFood != null) (closestFood.x - x) * INV_DETECTION_RADIUS else 0f
        val resourceDY = if (closestFood != null) (closestFood.y - y) * INV_DETECTION_RADIUS else 0f
        metaData[12] = resourceDX * -facingY + resourceDY * facingX
        metaData[13] = resourceDX * facingX + resourceDY * facingY
        val resourceDistance = if (closestFood != null)
            cheapDistance(this, closestFood) * INV_DETECTION_RADIUS else 0f
        metaData[14] = resourceDistance

        // Environmental
        val incubationMultiplier = if (INCUBATING) 2f else 1f
        val competitors = (familiarCount + unFamiliarCount) / GROUP_SIZE
        val resourceDensity = foodValue / (1f + competitors)

        val hunger = (1 - (ENERGY / MAX_ENERGY)).coerceIn(0f, 1f)
        val explorationPressure = (
                (1f - resourceDensity.coerceIn(0f, 1f)) * 0.75f +
                        (1f - familiarGroupSize.coerceIn(0f, 1f)) * 0.25f) *
                (1f - (dangerCenterThreat + unFamiliarThreat).coerceIn(0f, 1f))
        val immediateThreat = acuteScore.coerceIn(0f, 1f)
        val crowdingState = (((familiarCount + unFamiliarCount) - GROUP_SIZE) / GROUP_SIZE)
        val threatState = ((dangerCenterStrength * dangerCenterThreat) + immediateThreat).coerceIn(0f, 1f)
        val discomfortState = unFamiliarGroupSize * unFamiliarThreat
        val safetyScore = max(
            ((1f - familiarDistance) * familiarGroupSize) - (unFamiliarThreat * unFamiliarGroupSize), // social safety
            (1f - threatState) * (1f - explorationPressure)
        )  // environmental safety


        // Main signal evaluation
        val adrenaline = // Urgency/Intensity signal
            (threatState.coerceIn(0f, 1f) * (1 - safetyScore.coerceIn(0f, 1f))) +
                    (discomfortState.coerceIn(0f, 1f) * (1 - safetyScore.coerceIn(0f, 1f))) +
                    (crowdingState.coerceIn(0f, 1f) * (1f - resourceDensity.coerceIn(0f, 1f))) +
                    immediateThreat * incubationMultiplier + // Immediate threat, no scalar
                    hunger * (1f - resourceDensity.coerceIn(0f, 1f)) + // Hunger with no food around
                    (-targetEval).coerceAtLeast(0f) * (1f - targetDistance)

        val dopamine = // Appetite/Desire signal
            targetEval.coerceAtLeast(0f) * (1 - targetDistance) +
                    hunger * resourceDensity.coerceIn(0f, 1f) + // Hunger * resource abundance
                    explorationPressure.coerceIn(0f, 1f) * (1f - resourceDensity.coerceIn(0f, 1f)) +
                    familiarGroupSize * (1f - familiarDistance) * 0.5f * incubationMultiplier
        val serotonin = // Satisfaction/Content signal
            safetyScore.coerceIn(0f, 1f) * 0.35f +
                    (1f - hunger).coerceIn(0f, 1f) * 0.35f +
                    resourceDensity.coerceIn(0f, 1f) * 0.20f +
                    REPUTATION.coerceIn(0f, 1f) * 0.10f


        val sum = adrenaline + dopamine + serotonin
        val invSum = 1 / sum

        val adrNorm = if (sum > 1f) adrenaline * invSum else adrenaline
        val dopNorm = if (sum > 1f) dopamine * invSum else dopamine
        val serNorm = if (sum > 1f) serotonin * invSum else serotonin


        // ========= State Assignment Block =========
        // Main Signals
        state[0] = adrNorm.coerceIn(0f, 1f)
        state[1] = dopNorm.coerceIn(0f, 1f)
        state[2] = serNorm.coerceIn(0f, 1f)
        // same hue group coordination
        state[3] = familiarDistance.coerceIn(0f, 1f)
        state[4] = familiarGroupSize.coerceIn(0f, 1f)
        // other hue group coordination
        state[5] = unFamiliarThreat.coerceIn(0f, 1f)
        state[6] = unFamiliarGroupSize.coerceIn(0f, 1f)
        // renegade group coordination
        state[7] = dangerCenterThreat.coerceIn(0f, 1f)
        state[8] = dangerCenterStrength.coerceIn(0f, 1f)
        // resource coordination
        state[9] = resourceDistance.coerceIn(0f, 1f)
        // target coordination
        state[10] = targetDistance.coerceIn(0f, 1f)
        // target preference
        state[11] = if (target == null) 0f else if (target.hue == hue) 1f else -1f
        state[12] = if (target == null) 0f else if (target.isMale != isMale) 1f else -1f
        state[13] = if (target == null) 0f else if (target.REPUTATION < 0.5f) 1f else 0f
        state[14] = if (target == null) 0f else if (target.INCUBATING) -1f else 1f
        // global environmental && self
        state[15] = if (withinActionRadius) 1f else 0f
        state[16] = crowdingState.coerceIn(-1f, 1f)
        state[17] = threatState.coerceIn(-1f, 1f)
        state[18] = if (INCUBATING) 1f else 0f
        state[19] = REPUTATION.coerceIn(0f, 1f)
        state[20] = safetyScore.coerceIn(0f, 1f)
        state[21] = resourceDensity.coerceIn(0f, 1f)
        state[22] = explorationPressure.coerceIn(0f, 1f)
        state[23] = hunger.coerceIn(0f, 1f)
        state[24] = discomfortState.coerceIn(0f, 1f)

    }

    private fun generateIntentState(state: FloatArray) {

        val familiarCenterX = metaData[0]
        val familiarCenterY = metaData[1]
        val familiarDistance = metaData[2]

        val unFamiliarCenterX = metaData[3]
        val unFamiliarCenterY = metaData[4]
        val unFamiliarThreat = metaData[5]

        val dangerCenterX = metaData[6]
        val dangerCenterY = metaData[7]
        val dangerCenterThreat = metaData[8]

        val targetX = metaData[9]
        val targetY = metaData[10]
        val targetDistance = metaData[11]

        val resourceX = metaData[12]
        val resourceY = metaData[13]
        val resourceDistance = metaData[14]

        // Softmax-Argmax logic for Intent
        //          EXPLORE (0)
        // 5 ->   TARGET (1)
        // 6 ->   FOOD (2)
        // 7 -> FAMILIAR (3)
        // 8 -> UNFAMILIAR (4)
        // 9 -> DANGER

        var intentDistance = 1f
        var alignmentError = 0f

        val intentTarget = output[5]
        val intentFood = output[6]
        val intentFam = output[7]
        val intentUnFam = output[8]
        val intentDanger = output[9]

        var invalidTarget = false

        when (network.COMMITED_INTENT) {
            1 -> invalidTarget = targetDistance == 0f
            2 -> invalidTarget = resourceDistance == 0f
            3 -> invalidTarget = familiarDistance == 0f
            4 -> invalidTarget = unFamiliarThreat == 0f
        }

        if (network.COMMITED_INTENT != 0 && !invalidTarget) {

            // Intent translation
            var steerX = (intentTarget * targetX) + (intentFood * resourceX) +
                    (intentFam * familiarCenterX) +
                    (intentUnFam * unFamiliarCenterX) + (intentDanger * dangerCenterX)

            var steerY = (intentTarget * targetY) + (intentFood * resourceY) +
                    (intentFam * familiarCenterY) +
                    (intentUnFam * unFamiliarCenterY) + (intentDanger * dangerCenterY)

            val normalizedTargetDistance = targetDistance * 2f

            val wTarget = abs(intentTarget)
            val wFood = abs(intentFood)
            val wFam = abs(intentFam)
            val wUnFam = abs(intentUnFam)
            val wDanger = abs(intentDanger)
            val wSum = wTarget + wFood + wFam + wUnFam + wDanger

            intentDistance = if (wSum > 1e-4f) (
                    wTarget * (if (intentTarget >= 0f) normalizedTargetDistance else 1f - normalizedTargetDistance) +
                            wFood * (if (intentFood >= 0f) resourceDistance else 1f - resourceDistance) +
                            wFam * (if (intentFam >= 0f) familiarDistance else 1f - familiarDistance) +
                            wUnFam * (if (intentUnFam >= 0f) 1f - unFamiliarThreat else unFamiliarThreat) +
                            wDanger * (if (intentDanger >= 0f) 1f - dangerCenterThreat else dangerCenterThreat)
                    ) / wSum else 0f

            val steerMagnitude = cheapDistance(steerX, steerY)
            if (steerMagnitude > 0.0001f) {
                steerX /= steerMagnitude
                steerY /= steerMagnitude
            }

            alignmentError = 0.5f * (1 - steerY) * if (abs(steerX) > 0f) sign(steerX) else 1f

            // Clear old exploration target
            explorationTargetX = 0f
            explorationTargetY = 0f

        } else {
            // Exploration block

            // Out of range check
            val distance = cheapDistance(explorationTargetX - x, explorationTargetY - y)
            if (distance > detectionRadius) {
                explorationTargetX = 0f
                explorationTargetY = 0f
            }

            if (explorationTargetX == 0f && explorationTargetY == 0f) {
                val angle = rng.nextFloat(0.25f, 0.75f) * TABLE_SIZE
                val cos = cosTable[angle.toInt()]
                val sin = sinTable[angle.toInt()]
                explorationTargetX = x + (facingX * cos - facingY * sin) * 35f
                explorationTargetY = y + (facingX * sin + facingY * cos) * 35f
            }

            val explDX = (explorationTargetX - x) * INV_DETECTION_RADIUS
            val explDY = (explorationTargetY - y) * INV_DETECTION_RADIUS

            val explorationDist = cheapDistance(explDX, explDY)

            var explX = (explDX * -facingY + explDY * facingX) + (intentDanger * dangerCenterX)
            var explY = (explDX * facingX + explDY * facingY) + (intentDanger * dangerCenterY)
            val explorationMag = cheapDistance(explX, explY)

            if (explorationMag > 0.0001f) {
                explX /= explorationMag
                explY /= explorationMag
                alignmentError = 0.5f * (1 - explY) * if (abs(explX) > 0f) sign(explX) else 1f
            } else alignmentError = 0f

            intentDistance = (explorationDist + (abs(intentDanger) * dangerCenterThreat)) / (1f + abs(intentDanger))

            if (explorationDist < 0.1f) {
                // Target reached
                explorationTargetX = 0f
                explorationTargetY = 0f
            }
        }

        state[0] = alignmentError.coerceIn(-1f, 1f)
        state[1] = intentDistance.coerceIn(0f, 1f)

    }

    private fun cheapDistance(a: Agent, b: Entity): Float {
        // Rough Euclidean approximation using voodoo
        val dx = abs(a.x - b.x)
        val dy = abs(a.y - b.y)

        val maxD = max(dx, dy)
        val minD = min(dx, dy)

        return 0.96043384f * maxD + 0.39782473f * minD
    }

    private fun cheapDistance(dx: Float, dy: Float): Float {
        // Rough Euclidean approximation using voodoo
        val ax = abs(dx)
        val ay = abs(dy)

        val maxD = max(ax, ay)
        val minD = min(ax, ay)

        return 0.96043384f * maxD + 0.39782473f * minD
    }

    private fun action(kill: Float, mate: Float, eat: Float) {

        val choice = maxOf(kill, mate, eat)
        if (choice > network.ACTION_THRESHOLD) {
            when (choice) {

                eat -> {
                    val energyRatio = ENERGY / MAX_ENERGY
                    if (network.explorationSignal) {
                        network.valence += if (energyRatio <= 1f) {
                            (1f - energyRatio).coerceAtLeast(0.5f) * (1 - foodFieldDistance)
                        } else {
                            -(energyRatio - 1f) * 0.3f  // slight punishment for overeating, scaled down
                        }
                        return
                    }

                    val food = foodField ?: return

                    ENERGY += 10f
                    food.value--

                    val valence = if (energyRatio <= 1f) {
                        (1f - energyRatio).coerceAtLeast(0.5f) * (1 - foodFieldDistance)
                    } else {
                        -(energyRatio - 1f) * 0.3f  // slight punishment for overeating, scaled down
                    }
                    network.valence += valence


                }

                else -> {

                    if (TARGET == null) {
                        network.valence -= 0.1f
                        return
                    }
                    val target = TARGET as Agent
                    if (!target.enabled) return

                    when (choice) {
                        mate -> {

                            if (network.explorationSignal) {
                                if (withinActionRadius && isMale != target.isMale) {
                                    network.valence += 1.0f
                                } else network.valence -= 0.25f
                                return
                            }

                            Main.spawnAttempts.incrementAndGet()
                            if (withinActionRadius && isMale != target.isMale) {
                                if (isMale) { // is male
                                    if (!target.INCUBATING) {
                                        target.INCUBATING = true
                                        if (hue == target.hue) Main.spawnedWithOwnHue.incrementAndGet()
                                        else Main.spawnedWithOtherHue.incrementAndGet()
                                        target.INCUBATION_TIMER = 200f
                                        target.INCUBATION_MATERIAL =
                                            GeneticMaterial(network.extractWeightsForReproduction(), hue!!, isApathic)
                                    } else network.valence -= 0.2f
                                } else if (!INCUBATING) {
                                    INCUBATING = true
                                    if (hue == target.hue) Main.spawnedWithOwnHue.incrementAndGet()
                                    else Main.spawnedWithOtherHue.incrementAndGet()
                                    INCUBATION_TIMER = 200f
                                    INCUBATION_MATERIAL =
                                        GeneticMaterial(
                                            target.network.extractWeightsForReproduction(),
                                            hue!!,
                                            target.isApathic
                                        )
                                } else network.valence -= 0.2f
                                ENERGY -= 5f
                                targetCooldown = 50
                                network.valence += 1f
                                return
                            } else network.valence -= 0.2f
                            targetCooldown = 10
                            return
                        }

                        kill -> {

                            if (network.explorationSignal) {
                                if (withinActionRadius) {
                                    network.valence += (1f - (ENERGY/MAX_ENERGY))
                                    if (target.REPUTATION < 0.5f && !isApathic)
                                        network.valence += 1f - (target.REPUTATION * 2f)
                                } else network.valence -= 0.1f
                                return
                            }

                            Main.killAttempts.incrementAndGet()
                            if (withinActionRadius) {
                                if (target.hue == hue) {
                                    Main.killedOwnHue.incrementAndGet()
                                } else {
                                    Main.killedOtherHue.incrementAndGet()
                                }
                                kill(target)
                                clearTarget()
                                ENERGY += target.ENERGY * 0.5f
                                targetCooldown = 50
                                if (target.REPUTATION < 0.5f && !isApathic)
                                    network.valence += 1f - (target.REPUTATION * 2f)
                                REPUTATION -= target.REPUTATION * 0.5f
                                return
                            } else network.valence -= 0.1f
                            targetCooldown = 10
                        }
                    }
                }
            }
        }
    }

    private fun move(rotate: Float, drive: Float, alignmentError: Float) {
        val movement = drive.coerceIn(-1f, 1f) * (1 - abs(alignmentError))
        val rotation = rotate.coerceIn(-1f, 1f)
        val angleIndex = (((rotation + 1) / 2) * TABLE_SIZE).toInt().coerceIn(0, TABLE_SIZE - 1)
        val cosA = cosTable[angleIndex]
        val sinA = sinTable[angleIndex]

        val newFX = facingX * cosA - facingY * sinA
        val newFY = facingX * sinA + facingY * cosA

        facingX = newFX
        facingY = newFY

        var newX = x + (facingX * movement)
        var newY = y + (facingY * movement)

        if (newX >= 999.99f) newX -= 999.99f
        if (newX < 0f) newX += 999.99f
        if (newY >= 999.99f) newY -= 999.99f
        if (newY < 0f) newY += 999.99f

        // Occupancy check
        val gridX = newX.toInt()
        val gridY = newY.toInt()
        val oldGridX = x.toInt()
        val oldGridY = y.toInt()
        Simulation.occupancyGrid[oldGridX][oldGridY] = false // clear old occupancy

        if (Simulation.occupancyGrid[gridX][gridY]) { // If new position is occupied
            // Try X
            if (!Simulation.occupancyGrid[gridX][oldGridY]) {
                x = newX
            }
            // Try Y
            if (!Simulation.occupancyGrid[oldGridX][gridY]) {
                y = newY
            }
        } else {
            x = newX
            y = newY
        }
        Simulation.occupancyGrid[x.toInt()][y.toInt()] = true

    }

    private fun setTarget(target: Agent?) {
        if (TARGET != null) clearTarget()
        TARGET = target ?: return
        withinActionRadius = actionRadiusCheck(target)
        for (i in 0 until target.targetedBy.size) {
            if (target.targetedBy[i] == null) {
                target.targetedBy[i] = this
                return
            }
        }
        TARGET = null
    }

    fun clearTarget() {
        val target = TARGET ?: return
        for (i in 0 until target.targetedBy.size) {
            if (target.targetedBy[i] === this) {
                target.targetedBy[i] = null
                TARGET = null
                withinActionRadius = false
            }
        }
    }

    private fun lineOfSightCheck(target: Agent): Boolean {
        if (!target.enabled) return false

        val dx = target.x - x
        val dy = target.y - y
        val targetDistSq = dx * dx + dy * dy
        val targetRadiusSq = targetRadius * targetRadius
        if (targetDistSq > 0.000001f && targetDistSq <= targetRadiusSq) {
            val moveLenSq = facingX * facingX + facingY * facingY
            if (moveLenSq > 0.000001f) {
                val forward = facingX * dx + facingY * dy
                return forward > 0 && forward * forward > targetDistSq * moveLenSq * 0.25f
            } else return false
        } else return false
    }

    private fun actionRadiusCheck(target: Agent?): Boolean {
        if (target == null) return false
        if (!target.enabled) return false

        val dx = target.x - x
        val dy = target.y - y
        val targetDistSq = dx * dx + dy * dy
        val actionRadiusSq = actionRadius * actionRadius
        if (targetDistSq <= actionRadiusSq && lineOfSightCheck(target)) {
            return true
        } else return false
    }

    private fun kill(target: Agent) {
        // Death related logic
        if (!target.enabled) return
        target.enabled = false
        KILL_COUNT++
        managePopHueCounter(target.hue, false)
        Main.populationCounter.decrementAndGet()
        Simulation.occupancyGrid[target.x.toInt()][target.y.toInt()] = false // clear occupancy
    }

    private fun starvation() {
        enabled = false
        managePopHueCounter(hue, false)
        Main.populationCounter.decrementAndGet()
        Main.starvationCounter.incrementAndGet()
        Simulation.occupancyGrid[x.toInt()][y.toInt()] = false // clear occupancy
    }

    fun analysis(): String {
        return network.analysis(postState, output)
    }

    fun exportWeights(): Network.WeightsPackage {
        return network.extractWeightsForReproduction()
    }

    fun importWeights(weightsPackage: Network.WeightsPackage) {
        network.importWeights(weightsPackage)
    }

    fun resetState() {
        ENERGY = 100f
        enabled = true
        TARGET = null
        MATE_CONDITION = false
        withinActionRadius = false
        INCUBATING = false
        INCUBATION_MATERIAL = null
        INCUBATION_TIMER = 0f
        targetCooldown = 0
    }

    private fun updateProximity() {

        Arrays.fill(agentsInProximity, null)
        Arrays.fill(entitiesInProximity, null)
        AGENTS_PRESENCE_COUNT = 0
        ENTITIES_PRESENCE_COUNT = 0


        val gx = (x / Simulation.CELL_SIZE).toInt()
        val gy = (y / Simulation.CELL_SIZE).toInt()

        scan@
        for (x in gx - 1..gx + 1) {
            for (y in gy - 1..gy + 1) {
                if (x < 0 || y < 0 || x >= Simulation.CELLS_PER_ROW || y >= Simulation.CELLS_PER_ROW) continue
                val amt = Simulation.gridCellCount[x][y]
                for (z in 0 until amt) {
                    val other = Simulation.entityGrid[x][y][z] ?: continue
                    if (!other.enabled) continue
                    if (other == this) continue

                    val dx = this.x - other.x
                    val dy = this.y - other.y
                    val distSq = dx * dx + dy * dy

                    if (distSq < detectionRadius * detectionRadius) { // suggestions: different radii for different purposes
                        if (AGENTS_PRESENCE_COUNT < PRESENCE_CAP) {
                            if (other is Agent) {
                                agentsInProximity[AGENTS_PRESENCE_COUNT] = other
                                AGENTS_PRESENCE_COUNT++
                            }
                        }
                        if (ENTITIES_PRESENCE_COUNT < 5) {
                            if (other is Food) {
                                entitiesInProximity[ENTITIES_PRESENCE_COUNT] = other
                                ENTITIES_PRESENCE_COUNT++
                            }
                        }
                    }
                }
            }
        }
    }

}


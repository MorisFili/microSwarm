package derivative.code.microswarm.entity

import derivative.code.microswarm.Main
import derivative.code.microswarm.Simulation
import derivative.code.microswarm.managePopHueCounter
import derivative.code.microswarm.network.Network
import javafx.scene.paint.Color
import java.util.Arrays
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random


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
    val detectionRadius = 40f // Max radius = 50 in the current proximity scan logic
    val targetRadius = detectionRadius / 2f
    val actionRadius = targetRadius / 10f
    val PRESENCE_CAP = 50
    val GROUP_SIZE = PRESENCE_CAP / 3f
    val agentsInProximity = arrayOfNulls<Agent>(PRESENCE_CAP)
    val entitiesInProximity = arrayOfNulls<Entity>(5)
    var withinActionRadius = false
    var TARGET: Agent? = null
    val targetedBy = arrayOfNulls<Agent>(5)
    var resourceField: Resource? = null
    var MATE_CONDITION = false
    val preState = FloatArray(network.networkInput)
    val postState = FloatArray(network.networkInput)
    var facingX = 0f
    var facingY = 0f
    var output = FloatArray(network.networkOutput)

    // Social features
    var AGENTS_PRESENCE_COUNT = 0
    var ENTITIES_PRESENCE_COUNT = 0
    var CREDIT = 0f
    var isRenegade = false
    var KILL_COUNT = 0f

    // Economy calculations
    val reproductionCost = 200
    var targetCooldown = 0
    val creditHorizonMultiplier = 10f // How many actions the state should care about
    var creditDenominator = reproductionCost * creditHorizonMultiplier

    // Genetic traits
    var sex = false
    var apathic = false

    fun randomizeTraits() {
        sex = Random.nextFloat() < 0.5
        apathic = Random.nextFloat() < 0.1
    }

    init {
        randomizeTraits()
    }


    fun syncGenerateIntent() {
        if (targetCooldown > 0) {
            targetCooldown--
        }
        updateProximity()
        generateState(preState)
    }

    fun asyncFeedForward() {
        network.feedForward(preState, output)
    }

    fun syncPerformAction() {
        action(output[2], output[3], output[4])
    }

    fun asyncActionEvaluation() {
        generateMetaData()
        network.actionEvaluation()
    }

    fun syncMovement() {
        move(output[0], output[1])
    }

    fun asyncStateEvaluation() {
        generateState(postState)
        network.stateEvaluation(preState, postState)
        network.weightAdjustment()
    }


    fun analysis(): String {
        return network.analysis(postState)
    }

    fun exportWeights(): Network.WeightsPackage {
        return network.extractWeightsForReproduction()
    }

    fun importWeights(weightsPackage: Network.WeightsPackage) {
        network.importWeights(weightsPackage)
    }

    private fun generateState(state: FloatArray) {
        var sameCount = 0f
        var sameDX = 0f
        var sameDY = 0f

        var otherCount = 0f
        var otherDX = 0f
        var otherDY = 0f

        var renegadeGroupCount = 0f
        var renegadeStrength = 0f
        var renegadeDX = 0f
        var renegadeDY = 0f

        // Target clearing logic
        val targetRadiusSq = targetRadius * targetRadius
        if (targetCooldown > 0) clearTarget()

        var target = TARGET

        if (target != null && !target.enabled) clearTarget()
        if (target != null) {
            val dx = target.x - x
            val dy = target.y - y
            val targetDistSq = dx * dx + dy * dy
            if (targetDistSq > targetRadiusSq) clearTarget()
        }

        var closestTarget: Agent? = null
        var closestDistSq = 1000f
        val moveLenSq = facingX * facingX + facingY * facingY

        for (entity in agentsInProximity) {
            if (entity == null) continue
            if (!entity.enabled) continue
            val dx = entity.x - x
            val dy = entity.y - y
            val distSq = dx * dx + dy * dy

            if (entity.hue == hue) {
                sameCount++
                sameDX += dx
                sameDY += dy
            } else {
                otherCount++
                otherDX += dx
                otherDY += dy
            }

            if (entity.isRenegade) {
                renegadeGroupCount++
                renegadeDX += dx
                renegadeDY += dy
                renegadeStrength += entity.KILL_COUNT
            }

            // Target assignment
            if (TARGET != null) continue // Target already assigned
            if (distSq > 0.00001f && moveLenSq > 0.000001f &&
                distSq < targetRadiusSq && distSq < closestDistSq
            ) {
                val forward = facingX * dx + facingY * dy
                if (forward > 0 && forward * forward > distSq * moveLenSq * 0.25f) {
                    closestTarget = entity
                    closestDistSq = distSq
                }
            }
        }

        if (TARGET == null && closestTarget != null) {
            setTarget(closestTarget)
            target = TARGET
        }

        // Pick closest resource
        var closestResource: Resource? = null
        var closestDistSqResource = 1000f
        for (entity in entitiesInProximity) {
            if (entity is Resource) {
                if (!entity.enabled) continue
                val dx = entity.x - x
                val dy = entity.y - y
                val distSq = dx * dx + dy * dy

                if (distSq < closestDistSqResource) {
                    closestDistSqResource = distSq
                    closestResource = entity
                }
            }
        }

        // Resource radius check
        if (closestResource != null) {
            if (closestDistSqResource < 100f) {
                resourceField = closestResource
            } else resourceField = null
        } else resourceField = null


        var acuteScore = 0f
        var targetedByRenegadeCount = 0f
        var minTargetedDistance = 1000f
        var closestTargetedBy: Agent? = null
        for (i in 0 until targetedBy.size) {
            val other = targetedBy[i] ?: continue
            if (!isRenegade && other.isRenegade) {
                targetedByRenegadeCount++
            }
            acuteScore += 1f * other.KILL_COUNT
            val distance = cheapDistance(this, other)
            if (distance < minTargetedDistance) {
                minTargetedDistance = distance
                closestTargetedBy = other
            }
        }

        // ====== State calculations ======

        val sameHueCenterDX =
            if (sameCount > 0) (sameDX / sameCount) / detectionRadius else 0f

        val sameHueCenterDY =
            if (sameCount > 0) (sameDY / sameCount) / detectionRadius else 0f

        val sameHueDistance = if (sameCount > 0) {
            val avgDX = sameDX / sameCount
            val avgDY = sameDY / sameCount

            val dist = cheapDistance(avgDX, avgDY)

            (dist / detectionRadius)
        } else 0f


        val otherHueCenterDX =
            if (otherCount > 0) (otherDX / otherCount) / detectionRadius else 0f

        val otherHueCenterDY =
            if (otherCount > 0) (otherDY / otherCount) / detectionRadius else 0f

        val otherHueThreat = if (otherCount > 0) {
            val avgDX = otherDX / otherCount
            val avgDY = otherDY / otherCount

            val dist = cheapDistance(avgDX, avgDY)

            (1f - dist / detectionRadius) // Inverse distance
        } else 0f

        val sameHueGroupStrength = (sameCount / GROUP_SIZE)
        val otherHueGroupStrength = (otherCount / GROUP_SIZE)

        val renegadeCenterDX =
            if (renegadeGroupCount > 0) (renegadeDX / renegadeGroupCount) / detectionRadius else 0f
        val renegadeCenterDY =
            if (renegadeGroupCount > 0) (renegadeDY / renegadeGroupCount) / detectionRadius else 0f
        val renegadeDistance = if (renegadeGroupCount > 0) {
            val avgDX = renegadeDX / renegadeGroupCount
            val avgDY = renegadeDY / renegadeGroupCount

            val dist = cheapDistance(avgDX, avgDY)

            (1f - dist / detectionRadius) // Inverse distance
        } else 0f
        val renegadeGroupStrength = (renegadeStrength / renegadeGroupCount) / GROUP_SIZE

        val targetDX = if (target != null) (target.x - x) / targetRadius else 0f
        val targetDY = if (target != null) (target.y - y) / targetRadius else 0f
        val targetDistance = if (target != null) cheapDistance(this, target) / targetRadius else 0f

        val invDangerDistance = 1 - (minTargetedDistance / targetRadius)


        val totalCount = sameCount + otherCount
        val crowdingState = ((totalCount - GROUP_SIZE) / GROUP_SIZE)

        val resourceDX = if (closestResource != null) closestResource.x / detectionRadius else 0f
        val resourceDY = if (closestResource != null) closestResource.y / detectionRadius else 0f
        val resourceDistance = if (closestResource != null)
            cheapDistance(this, closestResource) / detectionRadius else 0f

        // Danger states
        val dangerState = if (isRenegade) {
            if (renegadeStrength == 0f) 1f else renegadeStrength * 1f - renegadeDistance
        } else renegadeStrength * renegadeDistance

        val safetyScore = if (!isRenegade) (1 - sameHueDistance) * sameHueGroupStrength
        else renegadeDistance * renegadeGroupStrength

        val fightOrFlight = if (acuteScore == 0f) {
            0f
        } else if (targetedByRenegadeCount < 2f || acuteScore <= safetyScore) { // Fight response
            invDangerDistance
        } else -invDangerDistance // Flight response

        if (fightOrFlight > 0.33f) { // Switch target to attacker if threatened
            setTarget(closestTargetedBy)
        }

        withinActionRadius = actionRadiusCheck(TARGET)

        // 1 -> neuron is on, casts vote to influence if state is true
        // 0 -> neuron is off, doesn't vote because state is false/not relevant
        // -1 -> neuron is on, contribute inversely when state is true

        // Latest = 29
        state[0] = 1f // Bias/drive state, always on

        // Directional symmetry states (-1f, 1f)
        state[1] = sameHueCenterDX.coerceIn(-1f, 1f)
        state[2] = sameHueCenterDY.coerceIn(-1f, 1f)
        state[5] = otherHueCenterDX.coerceIn(-1f, 1f)
        state[6] = otherHueCenterDY.coerceIn(-1f, 1f)
        state[11] = renegadeCenterDX.coerceIn(-1f, 1f)
        state[12] = renegadeCenterDY.coerceIn(-1f, 1f)
        state[20] = targetDX.coerceIn(-1f, 1f)
        state[21] = targetDY.coerceIn(-1f, 1f)
        state[18] = crowdingState.coerceIn(-1f, 1f) // Pulled towards a group of (PRESENCE_CAP / 3f)

        // Unilateral states
        state[3] = sameHueDistance.coerceIn(0f, 1f) // Sensory, no eval
        state[4] = sameHueGroupStrength.coerceIn(0f, 1f)
        state[7] = otherHueThreat.coerceIn(0f, 1f) // Sensory, no eval
        state[8] = otherHueGroupStrength.coerceIn(0f, 1f)
        state[13] = renegadeDistance.coerceIn(0f, 1f) // Sensory, no eval
        state[14] = renegadeStrength.coerceIn(0f, 1f)
        state[23] = targetDistance.coerceIn(0f, 1f)

        // Acute states
        state[19] = dangerState.coerceIn(0f, 1f)
        state[28] = fightOrFlight.coerceIn(-1f, 1f) // 1 = fight, -1 = flight
        //state[29] = 0f // rest/wander TBA


        // Self-sensory states
        state[10] = (CREDIT / creditDenominator).coerceIn(0f, 1f)
        state[17] = if (isRenegade) 1f else 0f

        // Target sensory
        state[9] = if (target == null) 0f else if (target.hue == hue) 1f else -1f
        state[15] = if (target == null) 0f else if (target.sex != sex) 1f else -1f
        state[16] = if (target == null) 0f else if (target.isRenegade) 1f else 0f
        state[24] = if (target == null) 0f else (target.CREDIT / creditDenominator).coerceIn(0f, 1f)
        state[22] = if (withinActionRadius) 1f else 0f

        // Resource sensory
        state[25] = resourceDX.coerceIn(-1f, 1f)
        state[26] = resourceDY.coerceIn(-1f, 1f)
        state[27] = resourceDistance.coerceIn(0f, 1f)

        // Metadata
        network.metaData[0] = CREDIT
        network.metaData[3] = if (isRenegade) 1f else 0f
    }

    private fun generateMetaData() {
        if (CREDIT > creditDenominator) creditDenominator = CREDIT
        network.metaData[1] = CREDIT
        network.metaData[2] = creditDenominator
    }

    private fun cheapDistance(a: Agent, b: Entity): Float {
        // Rough Euclidean approximation using magical numbers, ca 4.4% inaccuracy
        val dx = abs(a.x - b.x)
        val dy = abs(a.y - b.y)

        val maxD = max(dx, dy)
        val minD = min(dx, dy)

        return 0.96043384f * maxD + 0.39782473f * minD
    }

    private fun cheapDistance(dx: Float, dy: Float): Float {
        val ax = abs(dx)
        val ay = abs(dy)

        val maxD = max(ax, ay)
        val minD = min(ax, ay)

        return 0.96043384f * maxD + 0.39782473f * minD
    }

    private fun action(kill: Float, mate: Float, work: Float) {

        val choice = maxOf(kill, mate, work)
        if (choice > network.ACTION_THRESHOLD) {
            when (choice) {
                work -> {
                    if (resourceField != null) {

                        if (network.explorationSignal) {
                            network.valence += 1
                            return
                        }

                        val resource = resourceField ?: return
                        val creditGained = (1f + (PRESENCE_CAP / 3f) -
                                abs(AGENTS_PRESENCE_COUNT - (PRESENCE_CAP / 3f))).coerceAtLeast(1f)
                        CREDIT += creditGained
                        resource.value -= creditGained
                        return
                    }
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
                                if (withinActionRadius) {
                                    if (sex != target.sex) {
                                        network.valence += 0.5f
                                        if (target.hue == hue) {
                                            network.valence += 0.5f
                                        }
                                    } else network.valence -= 0.5f
                                } else network.valence -= 0.2f
                                return
                            }

                            Main.spawnAttempts.incrementAndGet()
                            if (sex != target.sex) {
                                if (CREDIT >= reproductionCost && withinActionRadius) {
                                    MATE_CONDITION = true
                                    CREDIT -= reproductionCost
                                    targetCooldown = 200
                                    if (hue == target.hue) {
                                        network.valence += 1f
                                        Main.spawnedWithOwnHue.incrementAndGet()
                                    } else {
                                        network.valence += 0.5f
                                        Main.spawnedWithOtherHue.incrementAndGet()
                                    }
                                }
                            }
                            targetCooldown = 10
                            return
                        }

                        kill -> {

                            if (network.explorationSignal) {
                                if (withinActionRadius) {
                                    if (target.hue == hue) {
                                        val creditGain = if (CREDIT > 0) (CREDIT + target.CREDIT) / CREDIT else 1f
                                        network.valence += (1 - creditGain) / 0.5f // 50% gain = 1f
                                        if (!apathic) network.valence -= 1f
                                    } else {
                                        val creditGain = if (CREDIT > 0) (CREDIT + target.CREDIT) / CREDIT else 1f
                                        network.valence += (1 - creditGain) / 0.5f // 50% gain = 1f
                                    }
                                    if (target.isRenegade != isRenegade) network.valence += 1f
                                } else network.valence -= 0.2f
                                return
                            }

                            Main.killAttempts.incrementAndGet()
                            if (withinActionRadius) {
                                if (target.hue == hue) {
                                    Main.killedOwnHue.incrementAndGet()
                                } else {
                                    Main.killedOtherHue.incrementAndGet()
                                }
                                CREDIT += target.CREDIT
                                target.CREDIT = 0f
                                kill(target)
                                clearTarget()
                                targetCooldown = 50
                                if (!target.isRenegade) isRenegade = true
                                if (target.isRenegade && !isRenegade) network.valence += 1f
                            }
                            targetCooldown = 10
                        }
                    }
                }
            }
        }
    }

    private fun move(nx: Float, ny: Float) {
        var newX = x + nx
        var newY = y + ny

        if (newX >= 999.99f) newX -= 999.99f
        if (newX < 0f) newX += 999.99f
        if (newY >= 999.99f) newY -= 999.99f
        if (newY < 0f) newY += 999.99f

        // Occupancy check
        val gridX = newX.toInt()
        val gridY = newY.toInt()
        Simulation.occupancyGrid[x.toInt()][y.toInt()] = false // clear old occupancy

        if (Simulation.occupancyGrid[gridX][gridY]) { // If new position is occupied
            // Try X
            if (!Simulation.occupancyGrid[gridX][y.toInt()]) {
                facingX = newX - x
                x = newX
            }
            // Try Y
            if (!Simulation.occupancyGrid[x.toInt()][gridY]) {
                facingY = newY - y
                y = newY
            }
        } else {
            facingX = newX - x
            facingY = newY - y
            x = newX
            y = newY
        }
        Simulation.occupancyGrid[x.toInt()][y.toInt()] = true

    }

    private fun setTarget(target: Agent?) {
        if (TARGET != null) clearTarget()
        TARGET = target ?: return
        for (i in 0 until target.targetedBy.size) {
            if (target.targetedBy[i] == null) {
                target.targetedBy[i] = this
                return
            }
        }
    }

    fun clearTarget() {
        val target = TARGET ?: return
        for (i in 0 until target.targetedBy.size) {
            if (target.targetedBy[i] === this) {
                target.targetedBy[i] = null
                TARGET = null
            }
        }
    }

    private fun actionRadiusCheck(target: Agent?): Boolean {
        if (target == null) return false
        if (!target.enabled) return false

        val dx = target.x - x
        val dy = target.y - y
        val targetDistSq = dx * dx + dy * dy
        val actionRadiusSq = actionRadius * actionRadius
        if (targetDistSq > 0.000001f && targetDistSq <= actionRadiusSq) {
            val moveLenSq = facingX * facingX + facingY * facingY
            if (moveLenSq > 0.000001f) {
                val forward = facingX * dx + facingY * dy
                if (forward > 0 && forward * forward > targetDistSq * moveLenSq * 0.25f) {
                    return true
                } else return false
            } else return false
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

    private fun updateProximity() {

        Arrays.fill(agentsInProximity, null)
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
                            if (other is Resource) {
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


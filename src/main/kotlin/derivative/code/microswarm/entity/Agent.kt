package derivative.code.microswarm.entity

import derivative.code.microswarm.Main
import derivative.code.microswarm.Simulation
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
    enabled: Boolean = true,
    private val network: Network
) : Entity(id, x, y, hue, enabled) {

    // Field variables
    val detectionRadius = 40f // Max radius = 50 in the current proximity scan logic
    val targetRadius = detectionRadius / 2f
    val actionRadius = targetRadius / 10f
    val PRESENCE_CAP = 50
    val agentsInProximity = arrayOfNulls<Agent>(PRESENCE_CAP)
    val entitiesInProximity = arrayOfNulls<Entity>(5)
    var withinActionRadius = false
    var TARGET: Entity? = null
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
    var RENEGADE = 0f

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


    fun asyncGenerateIntent() {
        if (RENEGADE > 0) RENEGADE -= 0.005f
        if (targetCooldown > 0) {
            TARGET = null
            targetCooldown--
        }
        updateProximity()
        generateState(preState)
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

        var renegadeCount = 0f
        var renegadeValue = 0f
        var renegadeDX = 0f
        var renegadeDY = 0f

        var societalCredit = 0f

        // If target is out of target range remove
        val targetRadiusSq = targetRadius * targetRadius
        if (TARGET != null) {
            val target = TARGET as Entity
            val dx = target.x - x
            val dy = target.y - y
            val targetDistSq = dx * dx + dy * dy
            if (targetDistSq > targetRadiusSq) TARGET = null // target lost
            if (!target.enabled) TARGET = null
        }

        var closestTarget: Entity? = null
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

            if (entity.RENEGADE > 0) {
                renegadeCount++
                renegadeDX += dx
                renegadeDY += dy
                renegadeValue += entity.RENEGADE
            }

            societalCredit += entity.CREDIT

            // Target assignment
            if (TARGET != null) continue
            if (targetCooldown > 0) continue
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

        if (TARGET == null) TARGET = closestTarget

        // Action radius check
        if (TARGET != null) {
            val target = TARGET as Entity
            val dx = target.x - x
            val dy = target.y - y
            val targetDistSq = dx * dx + dy * dy
            val actionRadiusSq = actionRadius * actionRadius
            if (targetDistSq > 0.000001f && targetDistSq <= actionRadiusSq) {
                val moveLenSq = facingX * facingX + facingY * facingY
                if (moveLenSq > 0.000001f) {
                    val forward = facingX * dx + facingY * dy
                    if (forward > 0 && forward * forward > targetDistSq * moveLenSq * 0.25f) {
                        withinActionRadius = true
                    } else withinActionRadius = false
                } else withinActionRadius = false
            } else withinActionRadius = false
        } else withinActionRadius = false

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


        //val avgSocietalCredit = if (sameCount + otherCount > 0) societalCredit / (sameCount + otherCount) else 0f

        // 1 -> neuron is on, casts vote to influence if state is true
        // 0 -> neuron is off, doesnt vote because state is false/not relevant
        // -1 -> neuron is on, contribute negatively when state is true

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

        val sameHueGroupStrength = (sameCount / (PRESENCE_CAP / 3f))
        val otherHueGroupStrength = (otherCount / (PRESENCE_CAP / 3f))

        val renegadeCenterDX =
            if (renegadeCount > 0) (renegadeDX / renegadeCount) / detectionRadius else 0f
        val renegadeCenterDY =
            if (renegadeCount > 0) (renegadeDY / renegadeCount) / detectionRadius else 0f
        val renegadeThreat = if (renegadeCount > 0) {
            val avgDX = renegadeDX / renegadeCount
            val avgDY = renegadeDY / renegadeCount

            val dist = cheapDistance(avgDX, avgDY)

            (1f - dist / detectionRadius) // Inverse distance
        } else 0f
        val renegadeStrength = renegadeValue / (PRESENCE_CAP / 3f)

        val target = TARGET as Agent?
        val targetDX = if (target != null) (target.x - x) / targetRadius else 0f
        val targetDY = if (target != null) (target.y - y) / targetRadius else 0f
        val targetDistance = if (target != null) cheapDistance(this, target) / targetRadius else 0f

        val totalCount = sameCount + otherCount
        val crowdingState = ((totalCount - (PRESENCE_CAP / 3f)) / (PRESENCE_CAP / 3f))
        val aloneState = -crowdingState

        val resourceDX = if (closestResource != null) closestResource.x / detectionRadius else 0f
        val resourceDY = if (closestResource != null) closestResource.y / detectionRadius else 0f
        val resourceDistance = if (closestResource != null)
            cheapDistance(this, closestResource) / detectionRadius else 0f

        // Latest = 27
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

        // Unilateral states
        state[3] = sameHueDistance.coerceIn(0f, 1f) // Sensory, no eval
        state[4] = sameHueGroupStrength.coerceIn(0f, 1f)
        state[7] = otherHueThreat.coerceIn(0f, 1f) // Sensory, no eval
        state[8] = otherHueGroupStrength.coerceIn(0f, 1f)
        state[13] = renegadeThreat.coerceIn(0f, 1f) // Sensory, no eval
        state[14] = renegadeStrength.coerceIn(0f, 1f)
        state[18] = crowdingState.coerceIn(0f, 1f)
        state[19] = aloneState.coerceIn(0f, 1f)
        state[23] = targetDistance.coerceIn(0f, 1f)

        // Self-sensory states
        state[10] = (CREDIT / creditDenominator).coerceIn(0f, 1f)
        state[17] = if (RENEGADE > 0) 1f else 0f

        // Target sensory
        state[9] = if (target == null) 0f else if (target.hue == hue) 1f else -1f
        state[15] = if (target == null) 0f else if (target.sex != sex) 1f else -1f
        state[16] = if (target == null) 0f else if (target.RENEGADE > 0) 1f else 0f
        state[24] = if (target == null) 0f else (target.CREDIT / creditDenominator).coerceIn(0f, 1f)
        state[22] = if (withinActionRadius) 1f else 0f

        // Resource sensory
        state[25] = resourceDX.coerceIn(-1f, 1f)
        state[26] = resourceDY.coerceIn(-1f, 1f)
        state[27] = resourceDistance.coerceIn(0f, 1f)

        // Metadata
        network.metaData[0] = CREDIT
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

                    if (TARGET == null) return
                    val target = TARGET as Agent

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
                                }
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
                                    if (target.RENEGADE > 0) network.valence += 1f
                                }
                                return
                            }

                            Main.killAttempts.incrementAndGet()
                            if (withinActionRadius) {
                                if (target.hue == hue) {
                                    RENEGADE += 50
                                    Main.killedOwnHue.incrementAndGet()
                                    CREDIT += target.CREDIT
                                    target.CREDIT = 0f
                                    kill(target)
                                    TARGET = null
                                    targetCooldown = 50
                                    Main.populationCounter.decrementAndGet()
                                } else {
                                    RENEGADE += 10f
                                    Main.killedOtherHue.incrementAndGet()
                                    CREDIT += target.CREDIT
                                    target.CREDIT = 0f
                                    kill(target)
                                    TARGET = null
                                    targetCooldown = 50
                                    Main.populationCounter.decrementAndGet()
                                }
                                if (target.RENEGADE > 0) network.valence += 1f
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

    private fun kill(target: Agent) {
        // Death related logic
        target.enabled = false
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


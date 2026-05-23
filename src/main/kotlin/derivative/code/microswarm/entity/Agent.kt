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
    val detectionRadius = 35f
    val PRESENCE_CAP = 50
    val inProximity = arrayOfNulls<Agent>(PRESENCE_CAP)
    val actionRadiusSq = 4f
    var withinActionRadius = false
    val targetRadiusSq = 200f
    var TARGET: Entity? = null
    var MATE_CONDITION = false
    val preState = FloatArray(network.networkInput)
    val postState = FloatArray(network.networkInput)
    var facingX = 0f
    var facingY = 0f
    var output = FloatArray(network.networkOutput)

    // Social features
    var PRESENCE_COUNT = 0
    var CREDIT = 0
    var RENEGADE = 0f

    // Economy calculations
    val reproductionCost = 200
    var refractoryPeriod = 200
    val creditHorizonMultiplier = 15f // How many actions the state value can store
    val creditDenominator = reproductionCost * creditHorizonMultiplier

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
        if (refractoryPeriod > 0) refractoryPeriod--
        updateProximity()
        generateState(preState)
        output = network.feedForward(preState)
    }

    fun syncPerformAction() {
        action(output[2], output[3], output[4])
        if (apathic) network.valence.coerceAtLeast(0f)
    }
    fun asyncActionEvaluation() {
        network.actionEvaluation()
    }
    fun syncMovement() {
        move(output[0], output[1])
    }
    fun asyncStateEvaluation() {
        generateState(postState)
        network.stateEvaluation(preState, postState)
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

        for (entity in inProximity) {
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
            if (targetDistSq > 0.000001f && targetDistSq <= actionRadiusSq) {
                val moveLenSq = facingX * facingX + facingY * facingY
                if (moveLenSq > 0.000001f) {
                    val forward = facingX * dx + facingY * dy
                    if (forward > 0 && forward * forward > targetDistSq * moveLenSq * 0.25f) {
                        withinActionRadius = true
                    } else withinActionRadius = false
                }
            }
        }

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

            (1f - dist / detectionRadius)
        } else 0f


        val otherHueCenterDX =
            if (otherCount > 0) (otherDX / otherCount) / detectionRadius else 0f

        val otherHueCenterDY =
            if (otherCount > 0) (otherDY / otherCount) / detectionRadius else 0f

        val otherHueDistance = if (otherCount > 0) {
            val avgDX = otherDX / otherCount
            val avgDY = otherDY / otherCount

            val dist = cheapDistance(avgDX, avgDY)

            (1f - dist / detectionRadius)
        } else 0f

        val sameHueGroupStrength = (sameCount / (PRESENCE_CAP / 3f))
        val otherHueGroupStrength = (otherCount / (PRESENCE_CAP / 3f))

        val renegadeCenterDX =
            if (renegadeCount > 0) (renegadeDX / renegadeCount) / detectionRadius else 0f
        val renegadeCenterDY =
            if (renegadeCount > 0) (renegadeDY / renegadeCount) / detectionRadius else 0f
        val renegadeDistance = if (renegadeCount > 0) {
            val avgDX = renegadeDX / renegadeCount
            val avgDY = renegadeDY / renegadeCount

            val dist = cheapDistance(avgDX, avgDY)

            (1f - dist / detectionRadius)
        } else 0f
        val renegadeStrength = renegadeValue / (PRESENCE_CAP / 3f)

        val target = TARGET as Agent?
        val targetDX = if (target != null) (target.x - x) / 14.14f else 0f
        val targetDY = if (target != null) (target.y - y) / 14.14f else 0f

        val totalCount = sameCount + otherCount
        val crowdingState = ((totalCount - (PRESENCE_CAP / 3f)) / (PRESENCE_CAP / 3f))
        val aloneState = -crowdingState

        // Latest = 22
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
        state[7] = otherHueDistance.coerceIn(0f, 1f) // Sensory, no eval
        state[8] = otherHueGroupStrength.coerceIn(0f, 1f)
        state[13] = renegadeDistance.coerceIn(0f, 1f) // Sensory, no eval
        state[14] = renegadeStrength.coerceIn(0f, 1f)
        state[18] = crowdingState.coerceIn(0f, 1f)
        state[19] = aloneState.coerceIn(0f, 1f)

        // Self-sensory states
        state[10] = (CREDIT / creditDenominator).coerceIn(0f, 1f)
        state[17] = if (RENEGADE > 0) 1f else 0f

        // Target sensory
        state[9] = if (target == null) 0f else if (target.hue == hue) 1f else -1f
        state[15] = if (target == null) 0f else if (target.sex != sex) 1f else -1f
        state[16] = if (target == null) 0f else if (target.RENEGADE > 0) 1f else 0f
        state[22] = if (withinActionRadius) 1f else 0f

    }

    private fun cheapDistance(a: Agent, b: Agent): Float {
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

        if (maxOf(kill, mate, work) > network.ACTION_THRESHOLD) {

            if (work >= mate && work >= kill) {
                CREDIT += 1 * PRESENCE_COUNT
                return
            }

            if (TARGET == null || !withinActionRadius) {
                if (!network.intentWithoutAction) network.valence -= 0.1f
                return
            }
            if (TARGET is Agent) {
                val target = TARGET as Agent
                if (mate >= kill) {
                    if (sex != target.sex) {
                        network.valence += 0.3f
                        if (CREDIT >= reproductionCost &&
                            refractoryPeriod == 0 &&
                            !network.intentWithoutAction) {

                            MATE_CONDITION = true
                            CREDIT -= reproductionCost
                            refractoryPeriod = 200
                            network.valence += 1f
                            TARGET = null
                        }

                    }
                } else {
                    network.valence -= 0.3f
                    if (target.hue == hue) {
                        if (!network.intentWithoutAction) {
                            network.valence -= 0.5f
                            RENEGADE += 10
                            Main.killedOwnHue.incrementAndGet()
                        }
                    } else {
                        if (!network.intentWithoutAction) {
                            RENEGADE += 2f
                            Main.killedOtherHue.incrementAndGet()
                        }
                    }
                    if (target.RENEGADE > 0) network.valence += 1f
                    if (!network.intentWithoutAction) {
                        CREDIT += target.CREDIT
                        target.CREDIT = 0
                        target.enabled = false
                        TARGET = null
                        Main.populationCounter.decrementAndGet()
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

    private fun updateProximity() {

        Arrays.fill(inProximity, null)
        PRESENCE_COUNT = 0

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
                        if (PRESENCE_COUNT < PRESENCE_CAP) {
                            inProximity[PRESENCE_COUNT] = other as Agent
                            PRESENCE_COUNT++
                        } else break@scan
                    }
                }
            }
        }
    }

}


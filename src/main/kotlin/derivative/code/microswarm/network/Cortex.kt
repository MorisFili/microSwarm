package derivative.code.microswarm.network

import derivative.code.microswarm.Main
import derivative.code.microswarm.Simulation.Companion.rng
import derivative.code.microswarm.entity.*
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sign


class Cortex(
    val inputStates: Int,
    val motionInputs: Int,
    val movementAxis: Int,
    val actionIntent: Int,
    val outputIntent: Int
) {

    // Fixed Variables
    val networkOutputs = movementAxis + actionIntent + outputIntent

    private val inputNeurons = inputStates
    private val activityNeurons = inputStates
    private val outputNeurons = (actionIntent + outputIntent) * 2
    private val intentNeurons = actionIntent + outputIntent
    val learningRate = 0.08f
    private val weightDecay = 0.99985f // For the deepest layer
    val maxMemory = 1.0f
    val COMMITMENT_GATE = 0.3f
    var committedSpatialIntent = 0
    var committedActionIntent = 0


    // Genetic intent
    private val weightsInput: Array<FloatArray> =
        Array(activityNeurons) { FloatArray(inputNeurons) }
    private val weightsIntent: Array<FloatArray> =
        Array(intentNeurons) { FloatArray(activityNeurons) }
    private val outputWeights: Array<FloatArray> =
        Array(outputNeurons) { FloatArray(intentNeurons) }
    private val activityBias: FloatArray = FloatArray(activityNeurons)
    private val intentBias: FloatArray = FloatArray(intentNeurons)


    // ============ LOCOMOTION ============
    val motorUnits = 8
    val sharePerUnit = 1f / 8f
    val ROTATION_INDEX = 0
    val locomotionNeurons = movementAxis * 2 * motorUnits
    val locomotionWeights: Array<FloatArray> =
        Array(locomotionNeurons) { FloatArray(motionInputs) }
    val locomotionMemory: Array<FloatArray> =
        Array(locomotionNeurons) { FloatArray(motionInputs) }
    val motorInputs = FloatArray(motionInputs)
    val locomotionOutput = FloatArray(locomotionNeurons)
    val motorContribution = FloatArray(locomotionNeurons)
    var movementCounter = 0f
    var accDistanceReward = 0f
    var accAlignmentReward = 0f
    var avgDistanceReward = 0f
    var avgAlignmentReward = 0f


    // Plastic weights
    private val memoryIn: Array<FloatArray> =
        Array(activityNeurons) { FloatArray(inputNeurons) }
    private val memoryIntent: Array<FloatArray> =
        Array(intentNeurons) { FloatArray(activityNeurons) }
    private val outputMemory: Array<FloatArray> =
        Array(outputNeurons) { FloatArray(intentNeurons) }
    private val outputIsAction = BooleanArray(outputNeurons) { false }

    // Rolling Variables
    private val stateInputs = FloatArray(inputNeurons)
    private val intent = FloatArray(intentNeurons)
    private val activity = FloatArray(activityNeurons)
    private val output = FloatArray(outputNeurons)
    private val outputContribution = FloatArray(outputNeurons)
    private val outputBucket = FloatArray(outputNeurons)
    private val intentContribution = FloatArray(intentNeurons)
    private val intentBucket = FloatArray(intentNeurons)
    private val activityContribution = FloatArray(activityNeurons)
    private var appliedReward = 0f
    var valence = 0f
    private var actionEvaluation = false
    var explorationSignal = false
    var explorationSignalForDebug = false
    private var bootstrappedIntent = 0
    private var bootstrapTimer = 0
    private var bootDuration = 50
    private var bootSignalStrength = 0.5f
    private var bootStrengthGate = 0.4f

    // Valence multipliers
    private var multiplier = 1f
    private var valenceAccumulator = 0f
    private var multiplierCounter = 0f
    private var accuValence = 0f
    private var avgValence = 0f

    // Neuromodulators
    var adrenaline = 0f
    var serotonin = 0f
    var dopamine = 0f

    // Helpers
    private val targetChoices = intArrayOf(1, 2, 4, 5)
    private val foodChoices = intArrayOf(3)
    private val allChoices = intArrayOf(1, 2, 3, 4, 5)
    private val noChoices = intArrayOf(-1)


    init {
        arrayBiasSeeder(activityBias)
        arrayBiasSeeder(intentBias)
        matrixBiasSeeder(weightsInput)
        matrixBiasSeeder(weightsIntent)
        matrixBiasSeeder(outputWeights, positiveOnly = true)
        locomotionSeeder(locomotionWeights)
        val actionIndices = actionIntent * 2
        for (i in 0 until actionIndices) { // Output indices that are action based
            outputIsAction[i] = true
        }
    }


    fun generateIntent(stateInputArray: FloatArray, outputArray: FloatArray) {

        for (i in 0 until inputNeurons) stateInputs[i] = stateInputArray[i] // Snapshot

        for (i in 0 until activityNeurons) {
            var sum = activityBias[i]
            for (j in 0 until inputNeurons) {
                val weight = weightsInput[i][j] + memoryIn[i][j]
                sum += weight * stateInputs[j]
            }
            // Leaky ReLU
            activity[i] = (if (sum > 0f) sum else sum * 0.25f).coerceIn(-0.25f, 1f)
        }

        for (i in 0 until intentNeurons) {
            var sum = intentBias[i]
            for (j in 0 until activityNeurons) {
                val weight = weightsIntent[i][j] + memoryIntent[i][j]
                sum += weight * activity[j]
            }
            intent[i] = (if (sum > 0f) sum else sum * 0.25f).coerceIn(-0.25f, 1f)
        }

        for (i in 0 until outputNeurons) {
            var sum = 0f
            for (j in 0 until intentNeurons) {
                val weightsOI = outputWeights[i][j] + outputMemory[i][j]
                sum += weightsOI * intent[j]
            }
            output[i] = max(0f, sum) // ReLU Interference
        }

        generateAction()
        generateSpatialIntent()
        generatePreferences()

        for (i in 0 until outputNeurons step 2) outputArray[movementAxis + i / 2] = output[i] - output[i + 1]

    }


    private val actionHabituation = FloatArray(6) { 1f }
    private fun generateAction() {
        // Softmax-Argmax-like logic for actions
        // 0,1 -> KILL
        // 2,3 -> MATE
        // 4,5 -> EAT
        // 6,7 -> SHARE
        // 8,9 -> STEAL
        explorationSignalForDebug = false

        val foodAvailable = stateInputs[RESOURCE_DIST] > 0f
        val withinActionRadius = stateInputs[WITHIN_ACTION_RADIUS] == 1f
        val availableChoices = if (foodAvailable && withinActionRadius) {
            allChoices
        } else if (withinActionRadius) {
            targetChoices
        } else if (foodAvailable) {
            foodChoices
        } else noChoices

        var explorationChoice = availableChoices.random()

        val explorationChoiceMag = when (explorationChoice) {
            1 -> output[0] + output[1]
            2 -> output[2] + output[3]
            3 -> output[4] + output[5]
            4 -> output[6] + output[7]
            5 -> output[8] + output[9]
            else -> 1f
        }

        if (explorationChoiceMag < bootStrengthGate) {

            when (explorationChoice) { // Exploratory signal injection
                1 -> {
                    output[0] = 0.5f
                    output[1] = 0f
                }

                2 -> {
                    output[2] = 0.5f
                    output[3] = 0f
                }

                3 -> {
                    output[4] = 0.5f
                    output[5] = 0f
                }

                4 -> {
                    output[6] = 0.5f
                    output[7] = 0f
                }

                5 -> {
                    output[8] = 0.5f
                    output[9] = 0f
                }
            }
        } else explorationChoice = -1


        val KILL = (output[0] - output[1]).coerceIn(0.01f, 1f) * actionHabituation[1]
        val MATE = (output[2] - output[3]).coerceIn(0.01f, 1f) * actionHabituation[2]
        val EAT = (output[4] - output[5]).coerceIn(0.01f, 1f) * actionHabituation[3]
        val SHARE = (output[6] - output[7]).coerceIn(0.01f, 1f) * actionHabituation[4]
        val STEAL = (output[8] - output[9]).coerceIn(0.01f, 1f) * actionHabituation[5]

        val CHOICE = maxOf(KILL, MATE, EAT, SHARE, STEAL)

        if (CHOICE >= COMMITMENT_GATE) {
            // Keep winner, flatten rest
            when (CHOICE) {
                KILL -> {
                    committedActionIntent = 1
                }

                MATE -> {
                    committedActionIntent = 2
                }

                EAT -> {
                    committedActionIntent = 3
                }

                SHARE -> {
                    committedActionIntent = 4
                }

                STEAL -> {
                    committedActionIntent = 5
                }
            }

        } else committedActionIntent = 0

        if (explorationChoice == committedActionIntent) {
            explorationSignal = true
            explorationSignalForDebug = true
        }

        for (a in 1..5) {
            val survivalBypass = a == 3 && stateInputs[HUNGER] > 0.4f
            actionHabituation[a] = when {
                survivalBypass -> 1f
                a == committedActionIntent -> (actionHabituation[a] * 0.99f).coerceAtLeast(0.3f)
                else -> (actionHabituation[a] + 0.002f).coerceAtMost(1f)
            }
        }

    }


    private val spatialHabituation = FloatArray(5) { 1f }
    private fun generateSpatialIntent() {
        // Softmax-Argmax logic for Spatial Intent
        //          EXPLORE (0)
        // 10,11 ->   TARGET (1)
        // 12,13 ->   FOOD (2)
        // 14,15 -> FAMILIAR (3)
        // 16,17 -> UNFAMILIAR (4)
        // 18,19 -> DANGER <-- stays non-flattened

        val hungerSq = if (stateInputs[HUNGER] > 0.4f) stateInputs[HUNGER] * stateInputs[HUNGER] else 0f
        val foodScarcity = if (stateInputs[RESOURCE_DENSITY] > 0) 0f else 1f // tweak later for real scarcity
        val explorationUrgency = hungerSq * foodScarcity

        val foodInProximity = stateInputs[RESOURCE_DIST] > 0
        val friendInProximity = stateInputs[FRIEND_DIST] > 0
        val unfamiliarInProximity = stateInputs[UNFAM_DIST] > 0
        val targetInProximity = friendInProximity || unfamiliarInProximity || stateInputs[HOSTILE_DIST] > 0


        val targetSignalStrength = abs(output[10]) + abs(output[11])
        val foodSignalStrength = abs(output[12]) + abs(output[13])
        val friendSignalStrength = abs(output[14]) + abs(output[15])
        val unfamiliarSignalStrength = abs(output[16]) + abs(output[17])

        if (bootstrapTimer == 0 && explorationUrgency == 0f) {

            bootstrappedIntent = rng.nextInt(1, 5)

            when (bootstrappedIntent) {
                1 -> {
                    if (targetSignalStrength < bootStrengthGate && targetInProximity) {
                        output[10] = bootSignalStrength
                        bootstrappedIntent = 1
                        bootstrapTimer = bootDuration
                    }
                }

                2 -> {
                    if (foodSignalStrength < bootStrengthGate && foodInProximity) {
                        output[12] = bootSignalStrength
                        bootstrappedIntent = 2
                        bootstrapTimer = bootDuration
                    }
                }

                3 -> {
                    if (friendSignalStrength < bootStrengthGate && friendInProximity) {
                        output[14] = bootSignalStrength
                        bootstrappedIntent = 3
                        bootstrapTimer = bootDuration
                    }
                }

                4 -> {
                    if (unfamiliarSignalStrength < bootStrengthGate && unfamiliarInProximity) {
                        output[16] = bootSignalStrength
                        bootstrappedIntent = 4
                        bootstrapTimer = bootDuration
                    }
                }
            }
        } else {
            when (bootstrappedIntent) {
                1 -> {
                    output[10] = bootSignalStrength
                    if (bootstrapTimer > 0) bootstrapTimer--
                }

                2 -> {
                    output[12] = bootSignalStrength
                    if (bootstrapTimer > 0) bootstrapTimer--
                }

                3 -> {
                    output[14] = bootSignalStrength
                    if (bootstrapTimer > 0) bootstrapTimer--
                }

                4 -> {
                    output[16] = bootSignalStrength
                    if (bootstrapTimer > 0) bootstrapTimer--
                }
            }
        }

        var targetSignal = (output[10] - output[11]).coerceIn(-1f, 1f) * (1 - explorationUrgency)
        var foodSignal = (output[12] - output[13]).coerceIn(-1f, 1f) * (1 + explorationUrgency)
        var friendSignal = (output[14] - output[15]).coerceIn(-1f, 1f) * (1 - explorationUrgency)
        var unfamiliarSignal = (output[16] - output[17]).coerceIn(-1f, 1f) * (1 - explorationUrgency)

        when (committedSpatialIntent) {
            1 -> targetSignal += 0.1f * sign(targetSignal)
            2 -> foodSignal += 0.1f * sign(foodSignal)
            3 -> friendSignal += 0.1f * sign(friendSignal)
            4 -> unfamiliarSignal += 0.1f * sign(unfamiliarSignal)
        }

        targetSignal *= spatialHabituation[1]
        foodSignal *= spatialHabituation[2]
        friendSignal *= spatialHabituation[3]
        unfamiliarSignal *= spatialHabituation[4]

        val choice = maxOf(
            abs(targetSignal),
            abs(foodSignal),
            abs(friendSignal),
            abs(unfamiliarSignal)
        )

        if (choice > COMMITMENT_GATE) {
            // Keep winner, flatten rest
            when (choice) {
                abs(targetSignal) -> {
                    committedSpatialIntent = 1
                }

                abs(foodSignal) -> {
                    committedSpatialIntent = 2
                }

                abs(friendSignal) -> {
                    committedSpatialIntent = 3
                }

                abs(unfamiliarSignal) -> {
                    committedSpatialIntent = 4
                }
            }
        } else {
            // Exploration
            committedSpatialIntent = 0
        }

        if (bootstrapTimer > 0 && committedSpatialIntent != bootstrappedIntent) bootstrapTimer = 0

        for (i in 1..4) {
            val starvingFoodBypass = i == 2 && stateInputs[HUNGER] > 0.4f
            spatialHabituation[i] = when {
                starvingFoodBypass -> 1f
                i == committedSpatialIntent -> (spatialHabituation[i] * 0.99f).coerceAtLeast(0.3f)
                else -> (spatialHabituation[i] + 0.002f).coerceAtMost(1f)
            }
        }


    }

    private fun generatePreferences() {
        // Softmax-Argmax logic for Spatial Intent
        //
        // 20,21 ->   SEX -> (1) OPPOSITE (-1) SAME
        // 22,23 ->   VALENCE -> (1) POSITIVE (-1) NEGATIVE
        // 24,25 ->   HEALTH -> (1) HIGHEST (-1) LOWEST


        // Noise gate
        val sexPref = output[20] - output[21]
        val valencePref = output[22] - output[23]
        val healthPref = output[24] - output[25]

        if (abs(sexPref) < 0.1f) {
            output[20] = 0f
            output[21] = 0f
        }
        if (abs(valencePref) < 0.1f) {
            output[22] = 0f
            output[23] = 0f
        }
        if (abs(healthPref) < 0.1f) {
            output[24] = 0f
            output[25] = 0f
        }

    }

    fun generateMovement(motorInputArray: FloatArray, outputArray: FloatArray) {
        Cerebellum.generateMovement(this, motorInputArray, outputArray)
    }


    fun actionEvaluation() {
        actionEvaluation = true

        if (abs(valence) >= 0.0001f) backpropContribution(valence)

        explorationSignal = false
        actionEvaluation = false
    }


    fun stateEvaluation(preState: FloatArray, postState: FloatArray, neurotransmitters: FloatArray) {

        val phasicAdr = neurotransmitters[0] - adrenaline
        val phasicDopa = neurotransmitters[1] - dopamine
        val phasicSero = neurotransmitters[2] - serotonin

        if (multiplierCounter > 5) {
            multiplier = valenceAccumulator / multiplierCounter
            multiplierCounter = 0f
            valenceAccumulator = 0f
        }

        val multiValue = if (multiplier > 0) (0.1f / multiplier).coerceIn(0.1f, 10f) else 1f

        // ===================== Adrenaline Path =====================


        // Decreasing discomfort
        val disc = evaluateSmallerBetter(preState[DISCOMFORT], postState[DISCOMFORT])
        valenceAccumulator += abs(disc)

        // Decreasing threat
        val thr = evaluateSmallerBetter(preState[THREAT], postState[THREAT])
        valenceAccumulator += abs(thr)


        val adrVal = ((disc + thr) * multiValue).coerceIn(-1f, 1f)
        if (abs(adrVal) > 0f) {
            backpropContribution(adrVal, phasicAdr)
        }


        // ===================== Dopamine Path =====================

        // Resource distance
        val res = evaluateDecreaseDistance(preState[RESOURCE_DIST], postState[RESOURCE_DIST])
        valenceAccumulator += abs(res)

        val dopVal = (res * multiValue).coerceIn(-1f, 1f)
        if (abs(dopVal) > 0f) {
            backpropContribution(dopVal, phasicDopa)
        }


        // ===================== Serotonin Path =====================

        // Decreasing hunger (eating)
        val hun = evaluateSmallerBetterSquared(preState[HUNGER], postState[HUNGER])
        valenceAccumulator += abs(hun)

        val serVal = (hun * multiValue).coerceIn(-1f, 1f)
        if (abs(serVal) > 0f) {
            backpropContribution(serVal, phasicSero)
        }

        adrenaline += (neurotransmitters[0] - adrenaline) * 0.15f
        dopamine += (neurotransmitters[1] - dopamine) * 0.08f
        serotonin += (neurotransmitters[2] - serotonin) * 0.05f

        multiplierCounter++
    }


    fun movementEvaluation(preMotorInputs: FloatArray, postMotorInputs: FloatArray, networkOutputs: FloatArray) {
        Cerebellum.movementEvaluation(this, preMotorInputs, postMotorInputs, networkOutputs)
    }


    var accuReward = 0f
    var accuCounter = 0
    var avgRewMag = 0f
    fun backpropContribution(valence: Float, neuromodulator: Float = 0f) {
        appliedReward = valence

        for (i in 0 until outputBucket.size) outputBucket[i] = 0f
        for (i in 0 until intentBucket.size) intentBucket[i] = 0f

        // Contribution calculation for pairs excluding movement axis
        for (i in 0 until outputNeurons step 2) {

            if (eligibilityGate(i)) continue

            val exploration = (explorationSignal || (bootstrapTimer > 0 && !actionEvaluation)) && i != 18

            val axisDirection =
                if (exploration) 1f else sign(output[i] - output[i + 1])
            if (axisDirection != 0f) {
                val contribution = if (exploration) 1f
                else calculateContribution(output[i].coerceIn(0f, 1f), skipGate = true)

                val contribution2 =
                    calculateContribution(output[i + 1].coerceIn(0f, 1f), skipGate = true)
                outputBucket[i] += appliedReward * contribution * axisDirection * (1 + neuromodulator)
                outputBucket[i + 1] += -appliedReward * contribution2 * axisDirection * (1 + neuromodulator)
            }
        }

        for (i in 0 until outputBucket.size) outputContribution[i] += outputBucket[i]

        for (j in 0 until intentNeurons) {
            var totalCorrection = 0f
            for (i in 0 until outputNeurons) {
                totalCorrection += outputBucket[i] * (outputWeights[i][j] + outputMemory[i][j])
            }
            intentBucket[j] += totalCorrection * calculateContribution(intent[j], 0.1f)
        }

        for (i in 0 until intentBucket.size) intentContribution[i] += intentBucket[i]


    for (j in 0 until activityNeurons) {
        val activityDPost = if (activity[j] > 0) 1f else 0.25f
        var totalCorrection = 0f
        for (i in 0 until intentNeurons) {
            totalCorrection += intentBucket[i] * (weightsIntent[i][j] + memoryIntent[i][j])
        }
        activityContribution[j] += totalCorrection * activityDPost *
                calculateContribution(activity[j], 0.1f)
    }

    accuCounter++
    accuReward += abs(valence)
    accuValence += valence
    if (accuCounter > 100)
    {
        avgRewMag = accuReward / accuCounter
        avgValence = accuValence / accuCounter
        accuCounter = 0
        accuReward = 0f
        accuValence = 0f
    }

    this.valence = 0f
}

private fun eligibilityGate(i: Int): Boolean {
    val isActionPair = outputIsAction[i]

    if (isActionPair != actionEvaluation) return true // Not updated this loop

    // 0,1 -> KILL (1)
    // 2,3 -> MATE (2)
    // 4,5 -> EAT (3)
    // 6,7 -> SHARE (4)
    // 8,9 -> STEAL (5)
    // 10,11 ->   TARGET (1)
    // 12,13 ->   FOOD (2)
    // 14,15 -> FAMILIAR (3)
    // 16,17 -> UNFAMILIAR (4)
    // 18,19 -> DANGER

    if (i in 0..9) { // Skip all but committed
        val actionCommitIndex = 1 + (i / 2)
        if (actionCommitIndex != committedActionIntent) return true
    }

    val isTargetSelection = i == 20 || i == 21 || i == 22 || i == 23 || i == 24 || i == 25
    if (isTargetSelection && committedSpatialIntent != 1) return true // Skip target selection when not chasing target

    if (i in 10..17) { // Skip intent that is not committed
        val committedIntent = (i - 8) / 2
        val committed = committedSpatialIntent == committedIntent
        val bootstrapping = bootstrapTimer > 0 && bootstrappedIntent == committedIntent
        if (!committed && !bootstrapping) return true
    }

    return false
}

var backPropTest = 0f
var intentTest = 0f
var activityTest = 0f
fun weightAdjustment() {

    // Output layer
    for (i in 0 until outputNeurons) {
        for (j in 0 until intentNeurons) {
            val adaptiveWeightDecay =
                adaptiveWeightDecay(outputMemory[i][j], 0.987f, 0.97f)
            val delta = outputContribution[i] * intent[j] * learningRate
            outputMemory[i][j] = (outputMemory[i][j] * adaptiveWeightDecay + delta)
                .coerceIn(0f, maxMemory)
        }
    }

    // Deep layers
    for (i in 0 until intentNeurons) {
        for (j in 0 until activityNeurons) {
            val adaptiveWeightDecay =
                adaptiveWeightDecay(memoryIntent[i][j], 0.9975f, 0.994f)
            val delta = intentContribution[i] * activity[j] * learningRate
            memoryIntent[i][j] = (memoryIntent[i][j] * adaptiveWeightDecay + delta)
                .coerceIn(-maxMemory, maxMemory)
        }
    }

    for (j in 0 until activityNeurons) {
        for (k in 0 until inputNeurons) {
            val adaptiveWeightDecay = weightDecay // 0.99985
            val delta = stateInputs[k] * activityContribution[j] * learningRate
            memoryIn[j][k] = (memoryIn[j][k] * adaptiveWeightDecay + delta)
                .coerceIn(-maxMemory, maxMemory)
        }
    }

    backPropTest = 0f
    intentTest = 0f
    activityTest = 0f
    // Post adjustment cleanup
    for (i in 0 until output.size) {
        backPropTest += abs(outputContribution[i])
        outputContribution[i] = 0f
    }
    for (i in 0 until intentContribution.size) {
        intentTest += abs(intentContribution[i])
        intentContribution[i] = 0f
    }
    for (i in 0 until activityContribution.size) {
        activityTest += abs(activityContribution[i])
        activityContribution[i] = 0f
    }

}


// ===================== HELPER FUNCTIONS ===================== \\

private fun evaluateSmallerBetter(
    preState: Float,
    postState: Float,
    noiseGate: Float = 0.001f,
    inverse: Boolean = false
): Float {
    val delta = (abs(preState) - abs(postState))
    if (abs(delta) < noiseGate) return 0f
    val valenceOut = delta * if (inverse) -1f else 1f // sign flip for negative

    return valenceOut
}

private fun evaluateSmallerBetterSquared(
    preState: Float,
    postState: Float,
    noiseGate: Float = 0.001f,
    inverse: Boolean = false
): Float {
    val preStateSq = preState * preState
    val postStateSq = postState * postState
    val delta = (abs(preStateSq) - abs(postStateSq))
    if (abs(delta) < noiseGate) return 0f
    val valenceOut = delta * if (inverse) -1f else 1f // sign flip for negative

    return valenceOut
}

fun evaluateRotation(
    alignmentError: Float,
    rotationOutput: Float
): Float {
    val rotation = rotationOutput.coerceIn(-1f, 1f)
    return abs(alignmentError) - abs(alignmentError - rotation)
}

fun evaluateDecreaseDistance(
    preState: Float,
    postState: Float,
    noiseGate: Float = 0.001f,
    inverse: Boolean = false
): Float {
    if (preState == 0f || postState == 0f) return 0f
    val delta = (abs(preState) - abs(postState))
    if (abs(delta) < noiseGate) return 0f
    val valenceOut = delta * if (inverse) -1f else 1f // sign flip for negative

    return valenceOut
}

fun adaptiveWeightDecay(memoryValue: Float, decay: Float, maxDecay: Float): Float {
    val memory = (abs(memoryValue) / maxMemory).coerceIn(0f, 1f)
    val pressure = (memory * memory).coerceIn(0f, 1f)
    return decay - ((decay - maxDecay) * pressure)
}

fun calculateContribution(
    activation: Float,
    eligibilityFloor: Float = 0.25f,
    skipGate: Boolean = false
): Float {
    if (abs(activation) <= 0.001f && !skipGate) return 0f
    val eligibility = eligibilityFloor + (1f - eligibilityFloor) * abs(activation)
    val plasticityGate = max(0.05f, 1f - eligibility * eligibility)

    return eligibility * plasticityGate
}


class WeightsPackage(
    val transferInput: Array<FloatArray>,
    val transferIntent: Array<FloatArray>,
    val transferOutput: Array<FloatArray>
)


fun extractWeightsForReproduction(): WeightsPackage {

    val exportInput = Array(activityNeurons) { i ->
        weightsInput[i].copyOf()
    }
    for (i in 0 until activityNeurons) {
        for (j in 0 until inputNeurons) {
            exportInput[i][j] = (exportInput[i][j] + memoryIn[i][j]) / 2
        }
    }

    val exportIntent = Array(intentNeurons) { i ->
        weightsIntent[i].copyOf()
    }
    for (i in 0 until intentNeurons) {
        for (j in 0 until activityNeurons) {
            exportIntent[i][j] = (exportIntent[i][j] + memoryIntent[i][j]) / 2
        }
    }

    val exportOutput = Array(outputNeurons) { i ->
        outputWeights[i].copyOf()
    }
    for (i in 0 until outputNeurons) {
        for (j in 0 until intentNeurons) {
            exportOutput[i][j] = (exportOutput[i][j] + outputMemory[i][j]) / 2
        }
    }

    return WeightsPackage(
        exportInput, exportIntent, exportOutput
    )
}

fun importWeights(weightsPackage: WeightsPackage) {
    val importInput = weightsPackage.transferInput
    for (i in 0 until weightsInput.size) {
        for (j in 0 until weightsInput[i].size) {
            weightsInput[i][j] = importInput[i][j]
        }
    }

    val importIntent = weightsPackage.transferIntent
    for (i in 0 until weightsIntent.size) {
        for (j in 0 until weightsIntent[i].size) {
            weightsIntent[i][j] = importIntent[i][j]
        }
    }

    val importOutput = weightsPackage.transferOutput
    for (i in 0 until outputWeights.size) {
        for (j in 0 until outputWeights[i].size) {
            outputWeights[i][j] = importOutput[i][j]
        }
    }


    resetMutable()
}

private fun resetMutable() {
    // Plastic weights
    for (i in 0 until memoryIn.size) {
        for (j in 0 until memoryIn[i].size) {
            memoryIn[i][j] = 0f
        }
    }
    for (i in 0 until memoryIntent.size) {
        for (j in 0 until memoryIntent[i].size) {
            memoryIntent[i][j] = 0f
        }
    }
    for (i in 0 until outputMemory.size) {
        for (j in 0 until outputMemory[i].size) {
            outputMemory[i][j] = 0f
        }
    }

    // Rolling variables
    valence = 0f
    actionEvaluation = false
    explorationSignal = false
}

private fun arrayBiasSeeder(array: FloatArray) {
    for (i in 0 until array.size) array[i] =
        rng.nextGaussian(0.0, 0.2).toFloat().coerceIn(-0.3f, 0.3f)
}

private fun matrixBiasSeeder(matrix: Array<FloatArray>, positiveOnly: Boolean = false) {
    val floor = if (positiveOnly) 0f else -0.2f
    for (i in 0 until matrix.size) {
        for (j in 0 until matrix[i].size) {
            matrix[i][j] = rng.nextGaussian(0.0, 0.1)
                .toFloat().coerceIn(floor, 0.2f)
        }
    }
}

private fun locomotionSeeder(matrix: Array<FloatArray>) {
    for (i in 0 until matrix.size) {
        for (j in 0 until matrix[i].size) {
            matrix[i][j] = rng.nextGaussian(0.0, 0.1)
                .toFloat().coerceIn(-0.2f, 0.2f)
            if (i >= (motorUnits * 2) && j == 1 && i % 2 == 0) {
                matrix[i][j] += 0.1f
            }
        }

    }
}


fun analysis(inputs: FloatArray, outputs: FloatArray): String {
    val block = StringBuilder(5000)
    block.append("===== START =====\n")
    block.append("Input State:\n")
    for (i in inputs.indices) {
        block.append("State(${i}): ${inputs[i]}\n")
    }
    block.append("Motion Inputs:\n")
    for (i in motorInputs.indices) {
        block.append("Motor(${i}): ${motorInputs[i]}\n")
    }
    if (Main.showAllWeights) {
        block.append("===== Memory weights: =====\n")
        block.append("Backpropagation Signal Strength\n")
        block.append("Layer(0): 100%\n")
        block.append("Layer(1): ${(intentTest / backPropTest) * 100f}%\n")
        block.append("Layer(2): ${(activityTest / backPropTest) * 100f}%\n")
        block.append("===== Input Neurons: =====\n")
        for (i in 0 until activityNeurons) {
            for (j in 0 until inputNeurons) {
                block.append("IN[${i}][${j}]")
                block.append(" Fixed: ${weightsInput[i][j]}")
                block.append("   Adapt: ${memoryIn[i][j]}\n")
            }
            block.append("Activity[${i}]: ${activity[i]}\n")
        }
        block.append("===== Intent Neurons: =====\n")
        for (i in 0 until intentNeurons) {
            for (j in 0 until activityNeurons) {
                block.append("Intent[${i}][${j}]")
                block.append(" Fixed: ${weightsIntent[i][j]}")
                block.append("   Adapt: ${memoryIntent[i][j]}\n")
            }
            block.append("Intent[${i}]: ${intent[i]}\n")
        }
        block.append("===== Ternary Neurons: =====\n")
        for (i in 0 until outputNeurons) {
            for (j in 0 until intentNeurons) {
                block.append("Output[${i}][${j}]")
                block.append(" Fixed: ${outputWeights[i][j]}")
                block.append("   Adapt: ${outputMemory[i][j]}\n")
            }
        }
    }
    block.append("===== Output Summary: =====\n")
    for (i in 0 until outputNeurons) {
        block.append("Output[${i}]: ${output[i]}\n")
    }

    if (Main.showAllWeights) {
        block.append("===== Motor Neurons: =====\n")
        for (i in 0 until locomotionNeurons) {
            for (j in 0 until motionInputs) {
                block.append("Motor [${i}][${j}]")
                block.append(" Fixed: ${locomotionWeights[i][j]}")
                block.append("   Adapt: ${locomotionMemory[i][j]}\n")
            }
        }
    }

    block.append("===== Motor Outputs:: =====\n")
    block.append("Rotation: ${outputs[0]}\n")
    block.append("Drive: ${outputs[1]}\n")

    block.append("===== Network Outputs:: =====\n")

    block.append("Kill: ${outputs[2]}\n")
    block.append("Mate: ${outputs[3]}\n")
    block.append("Eat: ${outputs[4]}\n")
    block.append("Share: ${outputs[5]}\n")
    block.append("Steal: ${outputs[6]}\n")
    block.append("Target Intent: ${outputs[7]}\n")
    block.append("Food Intent: ${outputs[8]}\n")
    block.append("Familiar Intent: ${outputs[9]}\n")
    block.append("Unfamiliar Intent: ${outputs[10]}\n")
    block.append("Danger Intent: ${outputs[11]}\n")

    block.append("===== Accessories: =====\n")
    block.append("AVG REWARD (100 ticks): $avgValence \n")
    block.append("AVG REWARD MAG (100 ticks): $avgRewMag \n")
    block.append("Avg. Movement Reward: $avgDistanceReward \n")
    block.append("===== Is currently: =====\n")
    if (committedActionIntent == 1 && explorationSignalForDebug) block.append("Thinking about Killing\n")
    if (committedActionIntent == 2 && explorationSignalForDebug) block.append("Thinking about Mating\n")
    if (committedActionIntent == 3 && explorationSignalForDebug) block.append("Thinking about Eating\n")
    if (committedActionIntent == 4 && explorationSignalForDebug) block.append("Thinking about Sharing\n")
    if (committedActionIntent == 5 && explorationSignalForDebug) block.append("Thinking about Stealing\n")
    if (committedActionIntent == 1 && !explorationSignalForDebug) block.append("Attempting to Kill\n")
    if (committedActionIntent == 2 && !explorationSignalForDebug) block.append("Attempting to Mate\n")
    if (committedActionIntent == 3 && !explorationSignalForDebug) block.append("Attempting to Eat\n")
    if (committedActionIntent == 4 && !explorationSignalForDebug) block.append("Attempting to Share\n")
    if (committedActionIntent == 5 && !explorationSignalForDebug) block.append("Attempting to Steal\n")
    if (bootstrapTimer > 0) {
        when (bootstrappedIntent) {
            1 -> block.append("Learning Target\n")
            2 -> block.append("Learning Food\n")
            3 -> block.append("Learning Familiar\n")
            4 -> block.append("Learning Unfamiliar\n")
        }
    }
    if (bootstrapTimer == 0) {
        when (committedSpatialIntent) {
            0 -> block.append("Exploring\n")
            1 -> if ((output[10] - output[11]) > 0)
                block.append("Chasing Target\n") else block.append("Fleeing from Target\n")

            2 -> if ((output[12] - output[13]) > 0)
                block.append("Chasing Food\n") else block.append("Fleeing from Food\n")

            3 -> if ((output[14] - output[15]) > 0)
                block.append("Chasing Friendly\n") else block.append("Fleeing from Friendly\n")

            4 -> if ((output[16] - output[17]) > 0)
                block.append("Chasing Unfamiliar\n") else block.append("Fleeing from Unfamiliar\n")
        }
    }
    if ((output[18] - output[19]) > 0)
        block.append("Approaching Unfriendly\n") else block.append("Fleeing from Unfriendly\n")

    block.append("===== Target Info: =====\n")
    if (committedSpatialIntent == 1) { // If target intent
        if (outputs[12] > 0) block.append("Opposite Sex\n")
        else if (outputs[12] < 0) block.append("Same Sex\n")
        else block.append("-----\n")
        if (outputs[13] > 0) block.append("Positive Valence\n")
        else if (outputs[13] < 0) block.append("Negative Valence\n")
        else block.append("-----\n")
        if (outputs[14] > 0) block.append("Highest Health\n")
        else if (outputs[14] < 0) block.append("Lowest Health\n")
        else block.append("-----\n")
    } else {
        block.append("-----\n")
        block.append("-----\n")
        block.append("-----\n")
    }
    block.append("===== END =====\n")
    return block.toString()
}


private val inputArrayClone = stateInputs.copyOf()
private val exActivity = FloatArray(activityNeurons)
private val exIntent = FloatArray(intentNeurons)
private val exDopaOutput = FloatArray(outputNeurons)

fun experimentalTargetAssessment(
    hue: Float,
    sex: Float,
    valence: Float,
    isIncubating: Float,
    reputation: Float,
    hunger: Float
): Float {

    stateInputs.copyInto(inputArrayClone)
    inputArrayClone[TARGET_HUE] = hue
    inputArrayClone[TARGET_SEX] = sex
    inputArrayClone[TARGET_VALENCE] = valence
    inputArrayClone[TARGET_IS_INCUB] = isIncubating
    inputArrayClone[TARGET_POPULARITY] = reputation
    inputArrayClone[TARGET_HUNGER] = hunger

    for (i in 0 until activityNeurons) {
        var sum = activityBias[i]
        for (j in 0 until inputNeurons) {
            val weight = weightsInput[i][j] + memoryIn[i][j]
            sum += weight * inputArrayClone[j]
        }
        // Leaky ReLU
        exActivity[i] = (if (sum > 0f) sum else sum * 0.25f).coerceIn(-0.25f, 1f)
    }

    for (i in 0 until intentNeurons) {
        var sum = intentBias[i]
        for (j in 0 until activityNeurons) {
            val weight = weightsIntent[i][j] + memoryIntent[i][j]
            sum += weight * exActivity[j]
        }
        exIntent[i] = max(0f, sum).coerceAtMost(1f)
    }

    for (i in 2..3) {
        var sumDopa = 0f
        for (j in 0 until intentNeurons) {
            val weightsOI = outputWeights[i][j] + outputMemory[i][j]
            sumDopa += weightsOI * exIntent[j]
        }
        exDopaOutput[i] = sumDopa
    }

    return exDopaOutput[2] - exDopaOutput[3]

}

}

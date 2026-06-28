package derivative.code.microswarm.network

import derivative.code.microswarm.Main
import derivative.code.microswarm.Simulation.Companion.rng
import derivative.code.microswarm.entity.State
import derivative.code.microswarm.modules.Action
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sign


class Cortex(
    val inputStates: Int,
    val motionInputs: Int,
    val movementAxis: Int,
    val outputPairs: Int,
    val outputPredictions: Int

) {

    // Fixed Variables
    val networkOutputs = movementAxis + (outputPairs * 2) + outputPredictions

    val inputNeurons = inputStates
    val activityNeurons = inputStates
    private val outputNeurons = (outputPairs * 2) + outputPredictions
    val intentNeurons = outputPairs
    var learningMultiplier = 1f
    val locomotionLearningRate = 0.06f
    val convergenceRate = 0.9999f
    val maxMemory = 2.0f
    val COMMITMENT_GATE = 0.2f
    var committedSpatialIntent = 0
    var committedActionIntent = 0
    var actionWeightGate = 0f

    // Decay rates
    private val INV_DECAY_SCALE = 1f / 0.3f // decay scale = magnitude of weight at which bigger decay kicks in
    val weightDecay = 0.9999f // For the deepest layer
    private val eligibilityConvergenceRate = 0.9f
    val outputLayerDecayBase = weightDecay
    val outputLayerDecayMax = 0.99f
    val intentLayerDecayBase = weightDecay
    val intentLayerDecayMax = 0.99f



    // Genetic intent
    val weightsInput: Array<FloatArray> =
        Array(activityNeurons) { FloatArray(inputNeurons) }
    val weightsIntent: Array<FloatArray> =
        Array(intentNeurons) { FloatArray(activityNeurons) }
    val outputWeights: Array<FloatArray> =
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
    val memoryIn: Array<FloatArray> =
        Array(activityNeurons) { FloatArray(inputNeurons) }
    private val eIn: Array<FloatArray> =
        Array(activityNeurons) { FloatArray(inputNeurons) }
    val memoryIntent: Array<FloatArray> =
        Array(intentNeurons) { FloatArray(activityNeurons) }
    private val eIntent: Array<FloatArray> =
        Array(intentNeurons) { FloatArray(activityNeurons) }
    val outputMemory: Array<FloatArray> =
        Array(outputNeurons) { FloatArray(intentNeurons) }
    private val eOut: Array<FloatArray> =
        Array(outputNeurons) { FloatArray(intentNeurons) }

    // Rolling Variables
    val stateInputs = FloatArray(inputNeurons)
    val intent = FloatArray(intentNeurons)
    val activity = FloatArray(activityNeurons)
    val output = FloatArray(outputNeurons)
    val outputContribution = FloatArray(outputNeurons)
    val intentContribution = FloatArray(intentNeurons)
    val activityContribution = FloatArray(activityNeurons)
    val actionOutputContribution = FloatArray(outputNeurons)
    val actionIntentContribution = FloatArray(intentNeurons)
    val actionActivityContribution = FloatArray(activityNeurons)
    var actionEvaluation = false
    private var bootstrappedIntent = 0
    private var bootstrappedPref = -1
    private var bootstrapTimer = 0
    private var bootDuration = 50
    private var bootSignalStrength = 0.5f
    var bootStrengthGate = 0.2f

    // Temporal
    val foodMemoryRot = FloatArray(3) { Float.NaN }
    val foodEntityId = IntArray(3) { -1 }
    var foodRingBuffer = 0

    // Neuromodulators
    var adrenaline = 0f
    var serotonin = 0f

    init {
        arrayBiasSeeder(activityBias)
        arrayBiasSeeder(intentBias)
        matrixBiasSeeder(weightsInput, 0.2f)
        matrixBiasSeeder(weightsIntent, 0.2f)
        outputSeeder(outputWeights)
        locomotionSeeder(locomotionWeights)
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
            activity[i] = (if (sum > 0f) sum else sum * 0.25f)
        }

        for (i in 0 until intentNeurons) {
            var sum = intentBias[i]
            for (j in 0 until activityNeurons) {
                val weight = weightsIntent[i][j] + memoryIntent[i][j]
                sum += weight * activity[j]
            }
            intent[i] = (if (sum > 0f) sum else sum * 0.25f)
        }

        for (i in 0 until CortexIndex.S_PREDICTION) {
            var sum = 0f
            for (j in 0 until intentNeurons) {
                val weightsOI = outputWeights[i][j] + outputMemory[i][j]
                sum += weightsOI * intent[j]
            }
            output[i] = max(0f, sum)
        }

        for (i in CortexIndex.S_PREDICTION until outputNeurons) {
            var sum = 0f
            for (j in 0 until intentNeurons) {
                val weightsOI = outputWeights[i][j] + outputMemory[i][j]
                sum += weightsOI * intent[j]
            }
            output[i] = sum
        }


        Action.Network.generateAction(this)
        generateSpatialIntent()
        generatePreferences()

        for (i in 0 until CortexIndex.S_PREDICTION step 2) outputArray[movementAxis + i / 2] = output[i] - output[i + 1]
        outputArray[OutputIndex.S_PREDICTION] = output[CortexIndex.S_PREDICTION]
        outputArray[OutputIndex.A_PREDICTION] = output[CortexIndex.A_PREDICTION]
    }


    val actionHabituation = FloatArray(6) { 1f }



    private val spatialHabituation = FloatArray(5) { 1f }
    private fun generateSpatialIntent() {
        // Softmax-Argmax logic for Spatial Intent
        //          EXPLORE (0)
        // 10,11 ->   TARGET (1)
        // 12,13 ->   FOOD (2)
        // 14,15 -> FAMILIAR (3)
        // 16,17 -> UNFAMILIAR (4)
        // 18,19 -> DANGER <-- stays non-flattened

        val hungerSq = if (stateInputs[State.Index.HUNGER] > 0.4f)
            stateInputs[State.Index.HUNGER] * stateInputs[State.Index.HUNGER] else 0f
        val foodScarcity = 1 - stateInputs[State.Index.RESOURCE_DENSITY]
        val explorationUrgency = hungerSq * foodScarcity

        val foodInProximity = stateInputs[State.Index.RESOURCE_DIST] > 0
        val friendInProximity = stateInputs[State.Index.FRIEND_DIST] > 0
        val unfamiliarInProximity = stateInputs[State.Index.UNFAM_DIST] > 0
        val targetInProximity = friendInProximity || unfamiliarInProximity || stateInputs[State.Index.HOSTILE_DIST] > 0


        val targetSignalStrength = abs(output[CortexIndex.S_TARGET]) + abs(output[CortexIndex.S_TARGET + 1])
        val foodSignalStrength = abs(output[CortexIndex.S_FOOD]) + abs(output[CortexIndex.S_FOOD + 1])
        val friendSignalStrength = abs(output[CortexIndex.S_FRIEND]) + abs(output[CortexIndex.S_FRIEND + 1])
        val unfamiliarSignalStrength = abs(output[CortexIndex.S_UNFAMILIAR]) +
                abs(output[CortexIndex.S_UNFAMILIAR + 1])

        if (bootstrapTimer == 0 && explorationUrgency < 0.25f) {

            val chance = rng.nextInt(1, 5)

            when (chance) {
                1 -> {
                    if (targetSignalStrength < bootStrengthGate && targetInProximity) {
                        output[CortexIndex.S_TARGET] = bootSignalStrength
                        bootstrappedPref = when (rng.nextInt(3)) {
                            0 -> 20; 1 -> 22; else -> 24
                        }
                        output[bootstrappedPref] = bootSignalStrength
                        bootstrappedIntent = 1
                        bootstrapTimer = bootDuration
                    }
                }

                2 -> {
                    if (foodSignalStrength < bootStrengthGate && foodInProximity) {
                        output[CortexIndex.S_FOOD] = bootSignalStrength
                        bootstrappedIntent = 2
                        bootstrapTimer = bootDuration
                    }
                }

                3 -> {
                    if (friendSignalStrength < bootStrengthGate && friendInProximity) {
                        output[CortexIndex.S_FRIEND] = bootSignalStrength
                        bootstrappedIntent = 3
                        bootstrapTimer = bootDuration
                    }
                }

                4 -> {
                    if (unfamiliarSignalStrength < bootStrengthGate && unfamiliarInProximity) {
                        output[CortexIndex.S_UNFAMILIAR] = bootSignalStrength
                        bootstrappedIntent = 4
                        bootstrapTimer = bootDuration
                    }
                }
            }
        } else {
            when (bootstrappedIntent) {
                1 -> {
                    output[CortexIndex.S_TARGET] = bootSignalStrength
                    output[bootstrappedPref] = bootSignalStrength
                    if (bootstrapTimer > 0) bootstrapTimer--
                }

                2 -> {
                    output[CortexIndex.S_FOOD] = bootSignalStrength
                    if (bootstrapTimer > 0) bootstrapTimer--
                }

                3 -> {
                    output[CortexIndex.S_FRIEND] = bootSignalStrength
                    if (bootstrapTimer > 0) bootstrapTimer--
                }

                4 -> {
                    output[CortexIndex.S_UNFAMILIAR] = bootSignalStrength
                    if (bootstrapTimer > 0) bootstrapTimer--
                }
            }
        }

        var targetSignal = (output[CortexIndex.S_TARGET] - output[CortexIndex.S_TARGET + 1]).coerceIn(-1f, 1f) *
                (1 - explorationUrgency) * spatialIntentSalienceFactor(Intents.TARGET)
        var foodSignal = (output[CortexIndex.S_FOOD] - output[CortexIndex.S_FOOD + 1]).coerceIn(-1f, 1f) *
                (1 + explorationUrgency) * spatialIntentSalienceFactor(Intents.FOOD)
        var friendSignal = (output[CortexIndex.S_FRIEND] - output[CortexIndex.S_FRIEND + 1]).coerceIn(-1f, 1f) *
                (1 - explorationUrgency) * spatialIntentSalienceFactor(Intents.FAMILIAR)
        var unfamiliarSignal =
            (output[CortexIndex.S_UNFAMILIAR] - output[CortexIndex.S_UNFAMILIAR + 1]).coerceIn(-1f, 1f) *
                    (1 - explorationUrgency) * spatialIntentSalienceFactor(Intents.UNFAMILIAR)

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
            val starvingFoodBypass = i == 2 && stateInputs[State.Index.HUNGER] > 0.4f
            spatialHabituation[i] = when {
                starvingFoodBypass -> 1f
                i == committedSpatialIntent -> (spatialHabituation[i] * 0.996f).coerceAtLeast(0.3f)
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
        val sexPref = output[CortexIndex.P_SEX] - output[CortexIndex.P_SEX + 1]
        val valencePref = output[CortexIndex.P_VALENCE] - output[CortexIndex.P_VALENCE + 1]
        val healthPref = output[CortexIndex.P_HEALTH] - output[CortexIndex.P_HEALTH + 1]

        if (abs(sexPref) < 0.1f) {
            output[CortexIndex.P_SEX] = 0f
            output[CortexIndex.P_SEX + 1] = 0f
        }
        if (abs(valencePref) < 0.1f) {
            output[CortexIndex.P_VALENCE] = 0f
            output[CortexIndex.P_VALENCE + 1] = 0f
        }
        if (abs(healthPref) < 0.1f) {
            output[CortexIndex.P_HEALTH] = 0f
            output[CortexIndex.P_HEALTH + 1] = 0f
        }

    }

    fun generateMovement(motorInputArray: FloatArray, outputArray: FloatArray) {
        Cerebellum.generateMovement(this, motorInputArray, outputArray)
    }


    fun actionEvaluation(neurotransmitters: FloatArray) {Action.Network.actionEvaluation(this, neurotransmitters)}

    var weightGate = 0.1f
    fun stateEvaluation(neurotransmitters: FloatArray) {



        val prediction = output[CortexIndex.S_PREDICTION]

        backpropContribution()

        // Calculate prediction error
        val error = ((neurotransmitters[1] - serotonin) - (neurotransmitters[0] - adrenaline)) - prediction

        adrenaline += (neurotransmitters[0] - adrenaline) * 0.15f
        serotonin += (neurotransmitters[1] - serotonin) * 0.05f


        if (abs(error) > weightGate) {
            predictionSignalUpdate(error, CortexIndex.S_PREDICTION)
            weightAdjustment(error)
            weightGate = abs(error)
        }
        weightGate = (weightGate - 0.001f).coerceAtLeast(0.05f)
    }


    fun movementEvaluation(preMotorInputs: FloatArray, postMotorInputs: FloatArray, steerCorrection: Float) {
        Cerebellum.movementEvaluation(this, preMotorInputs, postMotorInputs, steerCorrection)
    }

    fun backpropContribution() {

        // Contribution calculation for pairs excluding movement axis
        for (i in 0 until CortexIndex.S_PREDICTION step 2) {

            if (eligibilityGate(i)) continue

            val axisDirection = sign(output[i] - output[i + 1])
            if (axisDirection != 0f) {
                val contribution = calculateContribution(output[i].coerceIn(0f, 1f), skipGate = true)
                val contribution2 = calculateContribution(output[i + 1].coerceIn(0f, 1f), skipGate = true)
                outputContribution[i] += contribution * axisDirection
                outputContribution[i + 1] += -contribution2 * axisDirection
            }
        }


        for (j in 0 until intentNeurons) {
            val intentDPost = if (intent[j] > 0) 1f else 0.25f
            var totalCorrection = 0f
            for (i in 0 until outputNeurons) {
                totalCorrection += outputContribution[i] * (outputWeights[i][j] + outputMemory[i][j])
            }
            intentContribution[j] += totalCorrection * intentDPost *
                    calculateContribution(intent[j], 0.1f)
        }


        for (j in 0 until activityNeurons) {
            val activityDPost = if (activity[j] > 0) 1f else 0.25f
            var totalCorrection = 0f
            for (i in 0 until intentNeurons) {
                totalCorrection += intentContribution[i] * (weightsIntent[i][j] + memoryIntent[i][j])
            }
            activityContribution[j] += totalCorrection * activityDPost *
                    calculateContribution(activity[j], 0.1f)
        }

        // Neuron-level eligibility traces
        for (i in 0 until CortexIndex.S_PREDICTION) {
            if (eligibilityGate(i)) continue
            val post = outputContribution[i]
            for (j in 0 until intentNeurons)
                if (abs(eOut[i][j]) < 0.2f) {
                    eOut[i][j] += post * intent[j] * learningMultiplier
                } else eOut[i][j] = (eOut[i][j] * eligibilityConvergenceRate) +
                        (1f - eligibilityConvergenceRate) * post * intent[j] * learningMultiplier
        }
        for (i in 0 until intentNeurons) {
            val post = intentContribution[i]
            for (j in 0 until activityNeurons)
                if (abs(eIntent[i][j] ) < 0.2f) {
                    eIntent[i][j]  += post * activity[j] * learningMultiplier
                } else eIntent[i][j] = (eIntent[i][j] * eligibilityConvergenceRate) +
                        (1f - eligibilityConvergenceRate) * post * activity[j] * learningMultiplier

        }
        for (j in 0 until activityNeurons) {
            val post = activityContribution[j]
            for (k in 0 until inputNeurons)
                if (abs(eIn[j][k]) < 0.2f) {
                    eIn[j][k] += post * stateInputs[k] * learningMultiplier
                } else eIn[j][k] = (eIn[j][k] * eligibilityConvergenceRate) +
                        (1f - eligibilityConvergenceRate) * post * stateInputs[k] * learningMultiplier

        }

        learningMultiplier = 1f // reset to base

        for (i in 0 until output.size) outputContribution[i] = 0f
        for (i in 0 until intentContribution.size) intentContribution[i] = 0f
        for (i in 0 until activityContribution.size) activityContribution[i] = 0f


    }

    fun weightAdjustment(error: Float) {

        // Output layer
        for (i in 0 until CortexIndex.S_PREDICTION) {
            for (j in 0 until intentNeurons) {
                val adaptiveWeightDecay =
                    adaptiveWeightDecay(outputMemory[i][j],
                        outputLayerDecayBase, outputLayerDecayMax)
                val delta = eOut[i][j] * error
                outputMemory[i][j] = (outputMemory[i][j] * adaptiveWeightDecay + delta)
                    .coerceIn(0f, maxMemory)
                if (abs(outputMemory[i][j]) > 0f)
                    outputWeights[i][j] = (outputWeights[i][j] * convergenceRate) +
                            ((1 - convergenceRate) * outputMemory[i][j])
            }
        }

        // Deep layers
        for (i in 0 until intentNeurons) {
            for (j in 0 until activityNeurons) {
                val adaptiveWeightDecay =
                    adaptiveWeightDecay(memoryIntent[i][j],
                        intentLayerDecayBase, intentLayerDecayMax)
                val delta = eIntent[i][j] * error
                memoryIntent[i][j] = (memoryIntent[i][j] * adaptiveWeightDecay + delta)
                    .coerceIn(-maxMemory, maxMemory)
                if (abs(memoryIntent[i][j]) > 0f)
                    weightsIntent[i][j] = (weightsIntent[i][j] * convergenceRate) +
                            ((1 - convergenceRate) * memoryIntent[i][j])

            }
        }

        for (j in 0 until activityNeurons) {
            for (k in 0 until inputNeurons) {
                val adaptiveWeightDecay = weightDecay // 0.99985
                val delta = eIn[j][k] * error
                memoryIn[j][k] = (memoryIn[j][k] * adaptiveWeightDecay + delta)
                    .coerceIn(-maxMemory, maxMemory)
                if (abs(memoryIn[j][k]) > 0f)
                    weightsInput[j][k] = (weightsInput[j][k] * convergenceRate) +
                            ((1 - convergenceRate) * memoryIn[j][k])

            }
        }

        for (i in 0 until eOut.size) {
            for (j in 0 until eOut[i].size) {
                eOut[i][j] = 0f
            }
        }
        for (i in 0 until eIntent.size) {
            for (j in 0 until eIntent[i].size) {
                eIntent[i][j] = 0f
            }
        }
        for (i in 0 until eIn.size) {
            for (j in 0 until eIn[i].size) {
                eIn[i][j] = 0f
            }
        }

    }

    fun predictionSignalUpdate(error: Float, predictionType: Int) {


        val i = predictionType

        outputContribution[i] = 1f

        for (j in 0 until intentNeurons) {
            val intentDPost = if (intent[j] > 0) 1f else 0.25f
            val totalCorrection = outputContribution[i] * (outputWeights[i][j] + outputMemory[i][j])
            intentContribution[j] += totalCorrection * intentDPost *
                    calculateContribution(intent[j], 0.1f)
        }

        for (k in 0 until activityNeurons) {
            val activityDPost = if (activity[k] > 0) 1f else 0.25f
            var totalCorrection = 0f
            for (j in 0 until intentNeurons) {
                totalCorrection += intentContribution[j] * (weightsIntent[j][k] + memoryIntent[j][k])
            }
            activityContribution[k] += totalCorrection * activityDPost *
                    calculateContribution(activity[k], 0.1f)

        }



        for (j in 0 until intentNeurons) {
            val adaptiveWeightDecay =
                adaptiveWeightDecay(outputMemory[i][j],
                    outputLayerDecayBase, outputLayerDecayMax)
            val delta = outputContribution[i] * intent[j] * error
            outputMemory[i][j] = (outputMemory[i][j] * adaptiveWeightDecay + delta)
                .coerceIn(-maxMemory, maxMemory)
        }

        for (i in 0 until intentNeurons) {
            for (j in 0 until activityNeurons) {
                val adaptiveWeightDecay =
                    adaptiveWeightDecay(memoryIntent[i][j],
                        intentLayerDecayBase, intentLayerDecayMax)
                val delta = intentContribution[i] * activity[j] * error
                memoryIntent[i][j] = (memoryIntent[i][j] * adaptiveWeightDecay + delta)
                    .coerceIn(-maxMemory, maxMemory)
            }
        }

        for (j in 0 until activityNeurons) {
            for (k in 0 until inputNeurons) {
                val adaptiveWeightDecay = weightDecay // 0.99985
                val delta = activityContribution[j] * stateInputs[k] * error
                memoryIn[j][k] = (memoryIn[j][k] * adaptiveWeightDecay + delta)
                    .coerceIn(-maxMemory, maxMemory)
            }
        }

        for (i in 0 until output.size) outputContribution[i] = 0f
        for (i in 0 until intentContribution.size) intentContribution[i] = 0f
        for (i in 0 until activityContribution.size) activityContribution[i] = 0f
    }

    fun eligibilityGate(i: Int): Boolean {
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

        val isActionPair = i in 0..9

        if (isActionPair != actionEvaluation) return true // Not updated this loop

        if (i == CortexIndex.S_PREDICTION) return true // Updated separately
        if (i == CortexIndex.A_PREDICTION) return true // Updated separately

        if (i in 0..9) {
            val actionCommitIndex = 1 + (i / 2)
            if (actionCommitIndex != committedActionIntent) return true // Skip all but committed
            learningMultiplier = 10f
        }

        // Skip teaching target selection when not chasing target
        val isTargetSelection = i == 20 || i == 21 || i == 22 || i == 23 || i == 24 || i == 25
        if (isTargetSelection && committedSpatialIntent != 1) return true

        if (i in 10..17) { // Skip spatial intent that is not committed
            val spatialIntent = (i - 8) / 2
            if (spatialIntent != committedSpatialIntent) return true
            if (bootstrapTimer > 0 && bootstrappedIntent == spatialIntent) learningMultiplier = 10f
        }

        return false
    }


// ===================== HELPER FUNCTIONS ===================== \\

    fun spatialIntentSalienceFactor(intent: Int): Float {
        val max_amp = 0.3f //  30% signal amplification

        val hunger = stateInputs[State.Index.HUNGER] // hungry from 0 -> 1
        val threat = stateInputs[State.Index.THREAT] // threated from 0 -> 1
        val target_threat = stateInputs[State.Index.TARGET_THREAT]

        val value = when (intent) {
            Intents.KILL -> (target_threat * adrenaline) - serotonin
            Intents.MATE -> (1 - adrenaline) * (1 - threat)
            Intents.EAT -> (1 - adrenaline) * hunger
            Intents.SHARE -> serotonin
            Intents.STEAL -> (hunger * adrenaline) - ((1 - hunger) * serotonin)
            Intents.TARGET -> adrenaline * target_threat
            Intents.FOOD -> (hunger * (1 - adrenaline)) - ((1 - hunger) * serotonin)
            Intents.FAMILIAR -> serotonin
            Intents.UNFAMILIAR -> 1 - adrenaline
            else -> 0f
        }
        return 1f + value * max_amp
    }

    fun evaluateRotation(
        alignmentError: Float,
        rotationOutput: Float
    ): Float {
        return abs(alignmentError) - abs(alignmentError - rotationOutput)
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
        val memory = (memoryValue * INV_DECAY_SCALE).coerceIn(0f, 1f)
        val pressure = memory * memory * memory
        return decay + ((maxDecay - decay) * pressure)
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
        actionEvaluation = false
    }

    private fun arrayBiasSeeder(array: FloatArray) {
        for (i in 0 until array.size) array[i] =
            rng.nextGaussian(0.0, 0.2).toFloat().coerceIn(-0.3f, 0.3f)
    }

    private fun matrixBiasSeeder(matrix: Array<FloatArray>, floor: Float) {
        for (i in 0 until matrix.size) {
            for (j in 0 until matrix[i].size) {
                matrix[i][j] = rng.nextGaussian(0.0, 0.1)
                    .toFloat().coerceIn(-floor, floor)
            }
        }
    }

    private fun outputSeeder(outputMatrix: Array<FloatArray>) {
        for (i in 0 until outputNeurons) {
            val floor = when {
                i == CortexIndex.S_PREDICTION -> 0.2f
                i == CortexIndex.A_PREDICTION -> 0.2f
                else -> 0f
            }
            for (j in 0 until intentNeurons) {
                outputMatrix[i][j] = rng.nextGaussian(0.0, 0.1)
                    .toFloat().coerceIn(-floor, 0.2f)
            }
        }
    }

    private fun locomotionSeeder(matrix: Array<FloatArray>) {

        for (i in 0 until matrix.size) {
            val isDrive = i >= (motorUnits * 2)
            val isEven = i % 2 == 0
            for (j in 0 until matrix[i].size) {
                val isDistance = j == 1
                matrix[i][j] = if (!isDrive) {
                    if (isEven) rng.nextFloat(0.1f) else rng.nextFloat(-0.1f, 0f)
                } else rng.nextGaussian(0.1, 0.06).toFloat()
                if (isDrive && isDistance && isEven) {
                    matrix[i][j] += 0.1f
                }
            }

        }
    }


    fun analysis(inputs: FloatArray, outputs: FloatArray, neurotransmitters: FloatArray): String {
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
                block.append("Locomotion output: ${locomotionOutput[i]}\n")
            }
        }

        block.append("===== Motor Outputs:: =====\n")
        block.append("Rotation: ${outputs[OutputIndex.ROTATE]}\n")
        block.append("Drive: ${outputs[OutputIndex.DRIVE]}\n")

        block.append("===== Network Outputs:: =====\n")
        block.append("Truth: ${serotonin - adrenaline}\n")
        block.append("S_Prediction: ${outputs[OutputIndex.S_PREDICTION]}\n")
        block.append("S_Error: ${((neurotransmitters[1] - serotonin) - (neurotransmitters[0] - adrenaline)) - outputs[OutputIndex.S_PREDICTION]}\n")
        block.append("A_Prediction: ${outputs[OutputIndex.A_PREDICTION]}\n")
        block.append("A_Error: ${((neurotransmitters[1] - serotonin) - (neurotransmitters[0] - adrenaline)) - outputs[OutputIndex.A_PREDICTION]}\n")
        block.append("Serotonin: ${serotonin}\n")
        block.append("Adrenaline: ${adrenaline}\n")
        block.append("===== ================= =====\n")
        block.append("Kill: ${outputs[OutputIndex.KILL]}\n")
        block.append("Mate: ${outputs[OutputIndex.MATE]}\n")
        block.append("Eat: ${outputs[OutputIndex.EAT]}\n")
        block.append("Share: ${outputs[OutputIndex.SHARE]}\n")
        block.append("Steal: ${outputs[OutputIndex.STEAL]}\n")
        block.append("Target Intent: ${outputs[OutputIndex.TARGET]}\n")
        block.append("Food Intent: ${outputs[OutputIndex.FOOD]}\n")
        block.append("Familiar Intent: ${outputs[OutputIndex.FRIEND]}\n")
        block.append("Unfamiliar Intent: ${outputs[OutputIndex.UNFAMILIAR]}\n")
        block.append("Danger Intent: ${outputs[OutputIndex.UNFRIENDLY]}\n")
        block.append("===== Is currently: =====\n")
        if (committedActionIntent == 1) block.append("Attempting to Kill\n")
        if (committedActionIntent == 2) block.append("Attempting to Mate\n")
        if (committedActionIntent == 3) block.append("Attempting to Eat\n")
        if (committedActionIntent == 4) block.append("Attempting to Share\n")
        if (committedActionIntent == 5) block.append("Attempting to Steal\n")
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
                1 -> if ((output[CortexIndex.S_TARGET] - output[CortexIndex.S_TARGET + 1]) > 0)
                    block.append("Chasing Target\n") else block.append("Fleeing from Target\n")

                2 -> if ((output[CortexIndex.S_FOOD] - output[CortexIndex.S_FOOD + 1]) > 0)
                    block.append("Chasing Food\n") else block.append("Fleeing from Food\n")

                3 -> if ((output[CortexIndex.S_FRIEND] - output[CortexIndex.S_FRIEND + 1]) > 0)
                    block.append("Chasing Friendly\n") else block.append("Fleeing from Friendly\n")

                4 -> if ((output[CortexIndex.S_UNFAMILIAR] - output[CortexIndex.S_UNFAMILIAR + 1]) > 0)
                    block.append("Chasing Unfamiliar\n") else block.append("Fleeing from Unfamiliar\n")
            }
        }
        if ((output[18] - output[19]) > 0)
            block.append("Approaching Unfriendly\n") else block.append("Fleeing from Unfriendly\n")

        block.append("===== Target Info: =====\n")
        if (committedSpatialIntent == 1) { // If target intent
            if (outputs[OutputIndex.SEX] > 0) block.append("Opposite Sex\n")
            else if (outputs[OutputIndex.SEX] < 0) block.append("Same Sex\n")
            else block.append("-----\n")
            if (outputs[OutputIndex.VALENCE] > 0) block.append("Positive Valence\n")
            else if (outputs[OutputIndex.VALENCE] < 0) block.append("Negative Valence\n")
            else block.append("-----\n")
            if (outputs[OutputIndex.HEALTH] > 0) block.append("Highest Health\n")
            else if (outputs[OutputIndex.HEALTH] < 0) block.append("Lowest Health\n")
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
    private val matingOutput = FloatArray(outputNeurons)

    fun experimentalTargetAssessment(
        hue: Float,
        sex: Float,
        valence: Float,
        isIncubating: Float,
        reputation: Float,
        hunger: Float
    ): Float {

        stateInputs.copyInto(inputArrayClone)
        inputArrayClone[State.Index.TARGET_HUE] = hue
        inputArrayClone[State.Index.TARGET_SEX] = sex
        inputArrayClone[State.Index.TARGET_VALENCE] = valence
        inputArrayClone[State.Index.TARGET_IS_INCUB] = isIncubating
        inputArrayClone[State.Index.TARGET_POPULARITY] = reputation
        inputArrayClone[State.Index.TARGET_HUNGER] = hunger

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
            exIntent[i] = (if (sum > 0f) sum else sum * 0.25f).coerceIn(-0.25f, 1f)
        }

        for (i in 2..3) {
            var sum = 0f
            for (j in 0 until intentNeurons) {
                val weightsOI = outputWeights[i][j] + outputMemory[i][j]
                sum += weightsOI * exIntent[j]
            }
            matingOutput[i] = max(0f, sum)
        }

        return matingOutput[2] - matingOutput[3]

    }

}

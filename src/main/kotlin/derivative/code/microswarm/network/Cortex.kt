package derivative.code.microswarm.network

import derivative.code.microswarm.Main
import derivative.code.microswarm.Simulation.Companion.rng
import derivative.code.microswarm.entity.StateIndex
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sign


class Cortex(
    val inputStates: Int,
    val motionInputs: Int,
    val movementAxis: Int,
    val actionIntents: Int,
    val outputIntents: Int
) {

    // Fixed Variables
    val networkOutputs = movementAxis + actionIntents + outputIntents

    private val inputNeurons = inputStates
    private val activityNeurons = inputStates
    private val outputNeurons = (actionIntents + outputIntents) * 2
    private val intentNeurons = actionIntents + outputIntents
    val learningRate = 0.06f
    private val eligibilityDecay = 0.99f
    private val weightDecay = 0.9997f // For the deepest layer
    val maxMemory = 2.0f
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
    private val eIn: Array<FloatArray> =
        Array(activityNeurons) { FloatArray(inputNeurons) }
    private val memoryIntent: Array<FloatArray> =
        Array(intentNeurons) { FloatArray(activityNeurons) }
    private val eIntent: Array<FloatArray> =
        Array(intentNeurons) { FloatArray(activityNeurons) }
    private val outputMemory: Array<FloatArray> =
        Array(outputNeurons) { FloatArray(intentNeurons) }
    private val eOut: Array<FloatArray> =
        Array(outputNeurons) { FloatArray(intentNeurons) }
    private val outputIsAction = BooleanArray(outputNeurons) { false }

    // Rolling Variables
    private val stateInputs = FloatArray(inputNeurons)
    private val intent = FloatArray(intentNeurons)
    private val activity = FloatArray(activityNeurons)
    private val output = FloatArray(outputNeurons)
    private val outputContribution = FloatArray(outputNeurons)
    private val intentContribution = FloatArray(intentNeurons)
    private val activityContribution = FloatArray(activityNeurons)
    var valence = 0f
    private var actionEvaluation = false
    var explorationSignal = false
    var explorationSignalForDebug = false
    private var bootstrappedIntent = 0
    private var bootstrappedPref = -1
    private var bootstrapTimer = 0
    private var bootDuration = 50
    private var bootSignalStrength = 0.5f
    private var bootStrengthGate = 0.4f

    // Temporal
    val foodMemoryRot = FloatArray(3) { Float.NaN }
    val foodEntityId = IntArray(3) { -1 }
    var foodRingBuffer = 0

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
        val actionIndices = actionIntents * 2
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

        val foodAvailable = stateInputs[StateIndex.RESOURCE_DIST] > 0f
        val withinActionRadius = stateInputs[StateIndex.WITHIN_ACTION_RADIUS] == 1f
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


        val KILL = (output[0] - output[1]).coerceIn(0.01f, 1f) * actionHabituation[1] *
                getSalienceFactor(Intents.KILL)
        val MATE = (output[2] - output[3]).coerceIn(0.01f, 1f) * actionHabituation[2] *
                getSalienceFactor(Intents.MATE)
        val EAT = (output[4] - output[5]).coerceIn(0.01f, 1f) * actionHabituation[3] *
                getSalienceFactor(Intents.EAT)
        val SHARE = (output[6] - output[7]).coerceIn(0.01f, 1f) * actionHabituation[4] *
                getSalienceFactor(Intents.SHARE)
        val STEAL = (output[8] - output[9]).coerceIn(0.01f, 1f) * actionHabituation[5] *
                getSalienceFactor(Intents.STEAL)

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
            val survivalBypass = a == 3 && stateInputs[StateIndex.HUNGER] > 0.4f
            actionHabituation[a] = when {
                survivalBypass -> 1f
                a == committedActionIntent -> (actionHabituation[a] * 0.996f).coerceAtLeast(0.3f)
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

        val hungerSq = if (stateInputs[StateIndex.HUNGER] > 0.4f)
            stateInputs[StateIndex.HUNGER] * stateInputs[StateIndex.HUNGER] else 0f
        val foodScarcity = 1 - stateInputs[StateIndex.RESOURCE_DENSITY]
        val explorationUrgency = hungerSq * foodScarcity

        val foodInProximity = stateInputs[StateIndex.RESOURCE_DIST] > 0
        val friendInProximity = stateInputs[StateIndex.FRIEND_DIST] > 0
        val unfamiliarInProximity = stateInputs[StateIndex.UNFAM_DIST] > 0
        val targetInProximity = friendInProximity || unfamiliarInProximity || stateInputs[StateIndex.HOSTILE_DIST] > 0


        val targetSignalStrength = abs(output[10]) + abs(output[11])
        val foodSignalStrength = abs(output[12]) + abs(output[13])
        val friendSignalStrength = abs(output[14]) + abs(output[15])
        val unfamiliarSignalStrength = abs(output[16]) + abs(output[17])

        if (bootstrapTimer == 0 && explorationUrgency < 0.25f) {

            bootstrappedIntent = rng.nextInt(1, 5)

            when (bootstrappedIntent) {
                1 -> {
                    if (targetSignalStrength < bootStrengthGate && targetInProximity) {
                        output[10] = bootSignalStrength
                        bootstrappedPref = when (rng.nextInt(3)) { 0 -> 20; 1 -> 22; else -> 24 }
                        output[bootstrappedPref] = bootSignalStrength
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
                    output[bootstrappedPref] = bootSignalStrength
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

        var targetSignal = (output[10] - output[11]).coerceIn(-1f, 1f) *
                (1 - explorationUrgency) * getSalienceFactor(Intents.TARGET)
        var foodSignal = (output[12] - output[13]).coerceIn(-1f, 1f) *
                (1 + explorationUrgency) * getSalienceFactor(Intents.FOOD)
        var friendSignal = (output[14] - output[15]).coerceIn(-1f, 1f) *
                (1 - explorationUrgency) * getSalienceFactor(Intents.FAMILIAR)
        var unfamiliarSignal = (output[16] - output[17]).coerceIn(-1f, 1f) *
                (1 - explorationUrgency) * getSalienceFactor(Intents.UNFAMILIAR)

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
            val starvingFoodBypass = i == 2 && stateInputs[StateIndex.HUNGER] > 0.4f
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

        backpropContribution()

        explorationSignal = false
        actionEvaluation = false
    }


    fun stateEvaluation(preState: FloatArray, postState: FloatArray, neurotransmitters: FloatArray) {

        adrenaline += (neurotransmitters[0] - adrenaline) * 0.15f
        dopamine += (neurotransmitters[1] - dopamine) * 0.08f
        serotonin += (neurotransmitters[2] - serotonin) * 0.05f

        // Decreasing hunger (eating)
        val hunger = evaluateSmallerBetterSquared(preState[StateIndex.HUNGER], postState[StateIndex.HUNGER])

        if (abs(hunger) > 0f) {
            weightAdjustment(hunger)
        }
    }


    fun movementEvaluation(preMotorInputs: FloatArray, postMotorInputs: FloatArray, networkOutputs: FloatArray) {
        Cerebellum.movementEvaluation(this, preMotorInputs, postMotorInputs, networkOutputs)
    }

    fun backpropContribution() {

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

        for (i in 0 until outputNeurons) {
            val post = outputContribution[i]
            for (j in 0 until intentNeurons)
                eOut[i][j] = (eOut[i][j] * eligibilityDecay) + post * intent[j]
        }
        for (i in 0 until intentNeurons) {
            val post = intentContribution[i]
            for (j in 0 until activityNeurons)
                eIntent[i][j] = (eIntent[i][j] * eligibilityDecay) + post * activity[j]
        }
        for (j in 0 until activityNeurons) {
            val post = activityContribution[j]
            for (k in 0 until inputNeurons)
                eIn[j][k] = (eIn[j][k] * eligibilityDecay) + post * stateInputs[k]
        }

        for (i in 0 until output.size) outputContribution[i] = 0f
        for (i in 0 until intentContribution.size) intentContribution[i] = 0f
        for (i in 0 until activityContribution.size) activityContribution[i] = 0f


    }

    fun weightAdjustment(reward: Float) {

        // Output layer
        for (i in 0 until outputNeurons) {
            for (j in 0 until intentNeurons) {
                val adaptiveWeightDecay =
                    adaptiveWeightDecay(outputMemory[i][j], 0.987f, 0.97f)
                val delta = eOut[i][j] * learningRate * reward
                outputMemory[i][j] = (outputMemory[i][j] * adaptiveWeightDecay + delta)
                    .coerceIn(0f, maxMemory)
            }
        }

        // Deep layers
        for (i in 0 until intentNeurons) {
            for (j in 0 until activityNeurons) {
                val adaptiveWeightDecay =
                    adaptiveWeightDecay(memoryIntent[i][j], 0.9965f, 0.990f)
                val delta = eIntent[i][j] * learningRate * reward
                memoryIntent[i][j] = (memoryIntent[i][j] * adaptiveWeightDecay + delta)
                    .coerceIn(-maxMemory, maxMemory)
            }
        }

        for (j in 0 until activityNeurons) {
            for (k in 0 until inputNeurons) {
                val adaptiveWeightDecay = weightDecay // 0.99985
                val delta = eIn[j][k] * learningRate * reward
                memoryIn[j][k] = (memoryIn[j][k] * adaptiveWeightDecay + delta)
                    .coerceIn(-maxMemory, maxMemory)
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

        // Skip teaching target selection when not chasing target
        val isTargetSelection = i == 20 || i == 21 || i == 22 || i == 23 || i == 24 || i == 25
        if (isTargetSelection && committedSpatialIntent != 1) return true

        if (i in 10..17) { // Skip intent that is not committed
            val committedIntent = (i - 8) / 2
            val committed = committedSpatialIntent == committedIntent
            val bootstrapping = bootstrapTimer > 0 && bootstrappedIntent == committedIntent
            if (!committed && !bootstrapping) return true
        }

        return false
    }


// ===================== HELPER FUNCTIONS ===================== \\

    private fun getSalienceFactor(intent: Int): Float {
        val max_amp = 0.3f //  30% signal amplification

        val hunger = stateInputs[StateIndex.HUNGER] // hungry from 0 -> 1
        val threat = stateInputs[StateIndex.THREAT] // threated from 0 -> 1
        val safe = stateInputs[StateIndex.SAFETY_SCORE] // safe from 0 -> 1
        val exploration_pressure = stateInputs[StateIndex.EXP_PRESSURE]
        val target_threat = stateInputs[StateIndex.TARGET_THREAT]
        val target_attractive = stateInputs[StateIndex.TARGET_ATTRACTIVENESS]
        val target_attrN = target_attractive * 1.3333f

        val value = when (intent) {
            Intents.KILL -> (target_threat * adrenaline) - (safe * serotonin)
            Intents.MATE -> target_attrN * (dopamine - adrenaline) * (1 - threat)
            Intents.EAT -> (dopamine * hunger) - ((1 - hunger) * adrenaline)
            Intents.SHARE -> (1 - exploration_pressure) * serotonin - (exploration_pressure * dopamine)
            Intents.STEAL -> (hunger * adrenaline) - ((1 - hunger) * serotonin)
            Intents.TARGET -> max((dopamine * target_attractive), (adrenaline * target_threat)) - exploration_pressure
            Intents.FOOD -> (hunger * dopamine) - ((1 - hunger) * serotonin)
            Intents.FAMILIAR -> dopamine - exploration_pressure
            Intents.UNFAMILIAR -> (1 - adrenaline) * safe - exploration_pressure
            else -> 0f
        }
        return 1f + value * max_amp
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
        block.append("===== Neuromodulators: =====\n")
        block.append("Dopamine: $dopamine \n")
        block.append("Adrenaline: $adrenaline \n")
        block.append("Serotonin: $serotonin \n")
        block.append("===== Is currently: =====\n")
        block.append("===== Accessories: =====\n")
        block.append("\n")
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
        inputArrayClone[StateIndex.TARGET_HUE] = hue
        inputArrayClone[StateIndex.TARGET_SEX] = sex
        inputArrayClone[StateIndex.TARGET_VALENCE] = valence
        inputArrayClone[StateIndex.TARGET_IS_INCUB] = isIncubating
        inputArrayClone[StateIndex.TARGET_POPULARITY] = reputation
        inputArrayClone[StateIndex.TARGET_HUNGER] = hunger

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

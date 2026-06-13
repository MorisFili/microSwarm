package derivative.code.microswarm.network

import java.util.*
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sign


class Network(
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
    private val intentNeurons = inputStates / 2
    private val outputNeurons = (actionIntent + outputIntent) * 2
    private val learningRate = 0.08f
    private val weightDecay = 0.99985f // For the deepest layer
    private val maxMemory = 1.0f

    val ACTION_THRESHOLD = 0.5f
    val COMMITMENT_GATE = 0.15f
    var COMMITED_INTENT = 0

    // Util
    private val rng = Random()

    // Genetic intent
    private val weightsInput: Array<FloatArray> =
        Array(activityNeurons) { FloatArray(inputNeurons) }
    private val weightsIntent: Array<FloatArray> =
        Array(intentNeurons) { FloatArray(activityNeurons) }
    private val adrenalOutputWeights: Array<FloatArray> =
        Array(outputNeurons) { FloatArray(intentNeurons) }
    private val dopaminergicOutputWeights: Array<FloatArray> =
        Array(outputNeurons) { FloatArray(intentNeurons) }
    private val serotonergicOutputWeights: Array<FloatArray> =
        Array(outputNeurons) { FloatArray(intentNeurons) }
    private val activityBias: FloatArray = FloatArray(activityNeurons)
    private val intentBias: FloatArray = FloatArray(intentNeurons)


    // ============ LOCOMOTION ============
    private val motorUnits = 8
    private val sharePerUnit = 1f / 8f
    private val ROTATION_INDEX = 0
    private val locomotionNeurons = movementAxis * 2 * motorUnits
    private val locomotionWeights: Array<FloatArray> =
        Array(locomotionNeurons) { FloatArray(motionInputs) }
    private val locomotionMemory: Array<FloatArray> =
        Array(locomotionNeurons) { FloatArray(motionInputs) }
    private val motorInputs = FloatArray(motionInputs)
    private val locomotionOutput = FloatArray(locomotionNeurons)
    private val motorContribution = FloatArray(locomotionNeurons)


    // Plastic weights
    private val memoryIn: Array<FloatArray> =
        Array(activityNeurons) { FloatArray(inputNeurons) }
    private val memoryIntent: Array<FloatArray> =
        Array(intentNeurons) { FloatArray(activityNeurons) }
    private val adrenalOutputMemory: Array<FloatArray> =
        Array(outputNeurons) { FloatArray(intentNeurons) }
    private val dopaminergicOutputMemory: Array<FloatArray> =
        Array(outputNeurons) { FloatArray(intentNeurons) }
    private val serotonergicOutputMemory: Array<FloatArray> =
        Array(outputNeurons) { FloatArray(intentNeurons) }
    private val outputIsAction = BooleanArray(outputNeurons) { false }

    // Rolling Variables
    private var stateInputs = FloatArray(inputNeurons)
    private var intent = FloatArray(intentNeurons)
    private var activity = FloatArray(activityNeurons)
    private var adrenalOutput = FloatArray(outputNeurons)
    private var dopaminergicOutput = FloatArray(outputNeurons)
    private var serotonergicOutput = FloatArray(outputNeurons)
    private var outputSum = FloatArray(outputNeurons)
    private var adrenalContribution = FloatArray(outputNeurons)
    private var dopaminergicContribution = FloatArray(outputNeurons)
    private var serotonergicContribution = FloatArray(outputNeurons)
    private val adrenalBucket = FloatArray(outputNeurons)
    private val dopaBucket = FloatArray(outputNeurons)
    private val seroBucket = FloatArray(outputNeurons)
    private val intentContribution = FloatArray(intentNeurons)
    private val intentBucket = FloatArray(intentNeurons)
    private val activityContribution = FloatArray(activityNeurons)
    private var appliedReward = 0f
    var valence = 0f
    private var actionEvaluation = false
    var explorationSignal = false
    private var pendingActionBucket = 3
    private var bootstrappedIntent = 0
    private var bootstrapTimer = 0
    private var bootDuration = 50
    private var bootSignalStrength = 0.5f

    // Valence multipliers
    private var adrMagMulti = 1f
    private var dopMagMulti = 1f
    private var serMagMulti = 1f
    private var adrMagAcc = 0f
    private var dopMagAcc = 0f
    private var serMagAcc = 0f
    private var multiplierCounter = 0f
    private var accuValence = 0f
    private var avgValence = 0f


    init {
        arrayBiasSeeder(activityBias)
        arrayBiasSeeder(intentBias)
        matrixBiasSeeder(weightsInput)
        matrixBiasSeeder(weightsIntent)
        matrixBiasSeeder(adrenalOutputWeights, positiveOnly = true)
        matrixBiasSeeder(dopaminergicOutputWeights, positiveOnly = true)
        matrixBiasSeeder(serotonergicOutputWeights, positiveOnly = true)
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
            intent[i] = max(0f, sum).coerceAtMost(1f)
        }

        for (i in 0 until outputNeurons) {
            var sumAdr = 0f
            for (j in 0 until intentNeurons) {
                val weightsOI = adrenalOutputWeights[i][j] + adrenalOutputMemory[i][j]
                sumAdr += weightsOI * intent[j]
            }

            var sumDopa = 0f
            for (j in 0 until intentNeurons) {
                val weightsOI = dopaminergicOutputWeights[i][j] + dopaminergicOutputMemory[i][j]
                sumDopa += weightsOI * intent[j]
            }

            var sumSero = 0f
            for (j in 0 until intentNeurons) {
                val weightsOI = serotonergicOutputWeights[i][j] + serotonergicOutputMemory[i][j]
                sumSero += weightsOI * intent[j]
            }

            adrenalOutput[i] = sumAdr
            dopaminergicOutput[i] = sumDopa
            serotonergicOutput[i] = sumSero
            outputSum[i] = max(0f, sumAdr + sumDopa + sumSero) // ReLU Interference
        }

        // Softmax-Argmax-like logic for actions
        // 0,1 -> KILL
        // 2,3 -> MATE
        // 4,5 -> EAT

        val KILL = (outputSum[0] - outputSum[1]).coerceIn(0.01f, 1f)
        val MATE = (outputSum[2] - outputSum[3]).coerceIn(0.01f, 1f)
        val EAT = (outputSum[4] - outputSum[5]).coerceIn(0.01f, 1f)

        val CHOICE = maxOf(KILL, MATE, EAT)

        if (CHOICE < ACTION_THRESHOLD) {

            val ACTION_CHANCE = CHOICE / ACTION_THRESHOLD

            if (rng.nextFloat() < ACTION_CHANCE) {
                explorationSignal = true
                val CHANCE = (KILL + MATE + EAT) * rng.nextFloat()
                if (CHANCE < KILL) {
                    outputSum[0] = 1f
                    outputSum[1] = 0f
                    outputSum[2] = 0f
                    outputSum[3] = 0f
                    outputSum[4] = 0f
                    outputSum[5] = 0f
                    pendingActionBucket = 0
                } else if (CHANCE < KILL + MATE) {
                    outputSum[0] = 0f
                    outputSum[1] = 0f
                    outputSum[2] = 1f
                    outputSum[3] = 0f
                    outputSum[4] = 0f
                    outputSum[5] = 0f
                    pendingActionBucket = 1
                } else {
                    outputSum[0] = 0f
                    outputSum[1] = 0f
                    outputSum[2] = 0f
                    outputSum[3] = 0f
                    outputSum[4] = 1f
                    outputSum[5] = 0f
                    pendingActionBucket = 2
                }
            }

        } else if (CHOICE >= ACTION_THRESHOLD) {
            // Keep winner, flatten rest
            when (CHOICE) {
                KILL -> {
                    outputSum[2] = 0f
                    outputSum[3] = 0f
                    outputSum[4] = 0f
                    outputSum[5] = 0f
                    pendingActionBucket = 0
                }

                MATE -> {
                    outputSum[0] = 0f
                    outputSum[1] = 0f
                    outputSum[4] = 0f
                    outputSum[5] = 0f
                    pendingActionBucket = 1
                }

                EAT -> {
                    outputSum[0] = 0f
                    outputSum[1] = 0f
                    outputSum[2] = 0f
                    outputSum[3] = 0f
                    pendingActionBucket = 2
                }
            }

        }

        // Softmax-Argmax logic for Intent
        //          EXPLORE (0)
        // 6,7 ->   TARGET (1)
        // 8,9 ->   FOOD (2)
        // 10,11 -> FAMILIAR (3)
        // 12,13 -> UNFAMILIAR (4)
        // 14,15 -> DANGER

        if (bootstrapTimer == 0) {
            val intentTargetSignal = abs(outputSum[6]) + abs(outputSum[7])
            val intentFoodSignal = abs(outputSum[8]) + abs(outputSum[9])
            val intentFamSignal = abs(outputSum[10]) + abs(outputSum[11])
            val intentUnFamSignal = abs(outputSum[12]) + abs(outputSum[13])

            when (0f) {
                intentTargetSignal -> {
                    outputSum[6] = bootSignalStrength
                    bootstrappedIntent = 1
                    bootstrapTimer = bootDuration
                }

                intentFoodSignal -> {
                    outputSum[8] = bootSignalStrength
                    bootstrappedIntent = 2
                    bootstrapTimer = bootDuration
                }

                intentFamSignal -> {
                    outputSum[10] = bootSignalStrength
                    bootstrappedIntent = 3
                    bootstrapTimer = bootDuration
                }

                intentUnFamSignal -> {
                    outputSum[12] = bootSignalStrength
                    bootstrappedIntent = 4
                    bootstrapTimer = bootDuration
                }
            }
        } else {
            when (bootstrappedIntent) {
                1 -> {
                    outputSum[6] = bootSignalStrength
                    if (bootstrapTimer > 0) bootstrapTimer--
                }

                2 -> {
                    outputSum[8] = bootSignalStrength
                    if (bootstrapTimer > 0) bootstrapTimer--
                }

                3 -> {
                    outputSum[10] = bootSignalStrength
                    if (bootstrapTimer > 0) bootstrapTimer--
                }

                4 -> {
                    outputSum[12] = bootSignalStrength
                    if (bootstrapTimer > 0) bootstrapTimer--
                }
            }
        }

        var intentTarget = (outputSum[6] - outputSum[7]).coerceIn(-1f, 1f)
        var intentFood = (outputSum[8] - outputSum[9]).coerceIn(-1f, 1f)
        var intentFam = (outputSum[10] - outputSum[11]).coerceIn(-1f, 1f)
        var intentUnFam = (outputSum[12] - outputSum[13]).coerceIn(-1f, 1f)

        when (COMMITED_INTENT) {
            1 -> intentTarget += 0.1f * sign(intentTarget)
            2 -> intentFood += 0.1f * sign(intentFood)
            3 -> intentFam += 0.1f * sign(intentFam)
            4 -> intentUnFam += 0.1f * sign(intentUnFam)
        }

        if (bootstrapTimer > 0 && COMMITED_INTENT != bootstrappedIntent) bootstrapTimer = 0

        val choice = maxOf(
            abs(intentTarget),
            abs(intentFood),
            abs(intentFam),
            abs(intentUnFam))

        if (choice > COMMITMENT_GATE) {
            // Keep winner, flatten rest
            when (choice) {
                abs(intentTarget) -> {
                    outputSum[8] = 0f
                    outputSum[9] = 0f
                    outputSum[10] = 0f
                    outputSum[11] = 0f
                    outputSum[12] = 0f
                    outputSum[13] = 0f
                    COMMITED_INTENT = 1
                }

                abs(intentFood) -> {
                    outputSum[6] = 0f
                    outputSum[7] = 0f
                    outputSum[10] = 0f
                    outputSum[11] = 0f
                    outputSum[12] = 0f
                    outputSum[13] = 0f
                    COMMITED_INTENT = 2
                }

                abs(intentFam) -> {
                    outputSum[6] = 0f
                    outputSum[7] = 0f
                    outputSum[8] = 0f
                    outputSum[9] = 0f
                    outputSum[12] = 0f
                    outputSum[13] = 0f
                    COMMITED_INTENT = 3
                }

                abs(intentUnFam) -> {
                    outputSum[6] = 0f
                    outputSum[7] = 0f
                    outputSum[8] = 0f
                    outputSum[9] = 0f
                    outputSum[10] = 0f
                    outputSum[11] = 0f
                    COMMITED_INTENT = 4
                }
            }
        } else {
            // Exploration
            outputSum[6] = 0f
            outputSum[7] = 0f
            outputSum[8] = 0f
            outputSum[9] = 0f
            outputSum[10] = 0f
            outputSum[11] = 0f
            outputSum[12] = 0f
            outputSum[13] = 0f
            COMMITED_INTENT = 0
        }



        for (i in 0 until outputNeurons step 2) outputArray[movementAxis + i / 2] = outputSum[i] - outputSum[i + 1]

    }

    fun generateMovement(motorInputArray: FloatArray, outputArray: FloatArray) {
        for (i in 0 until motionInputs) motorInputs[i] = motorInputArray[i]

        // Locomotion block
        for (i in 0 until locomotionNeurons) {
            var sum = 0f
            val isDrive = i >= (motorUnits * 2) // First half of locomotion neurons is rotation, second half is drive
            for (j in 0 until motionInputs) {
                if (!isDrive && j != ROTATION_INDEX) continue // Distance shouldn't affect rotation
                val input = if (isDrive && j == ROTATION_INDEX)
                    abs(motorInputs[j]) else motorInputs[j] // Rotation error sign shouldn't affect drive
                val weight = locomotionWeights[i][j] + locomotionMemory[i][j]
                sum += weight * input
            }
            val torque = (((i / 2) % 8) * sharePerUnit) + sharePerUnit + 1f
            locomotionOutput[i] = max(0f, sum * torque).coerceAtMost(1f)
        }


        for (axis in 0 until movementAxis) {
            val base = axis * motorUnits * 2
            var positive = 0f
            var negative = 0f
            for (gear in 0 until motorUnits) {
                positive += locomotionOutput[base + (gear * 2)]
                negative += locomotionOutput[base + (gear * 2) + 1]
            }
            if (axis != ROTATION_INDEX && positive < 0.001f && negative < 0.001f && rng.nextFloat() < 0.1f) {
                locomotionOutput[base] = rng.nextFloat() * sharePerUnit
                locomotionOutput[base + 1] = rng.nextFloat() * sharePerUnit
                positive = locomotionOutput[base]
                negative = locomotionOutput[base + 1]
            }

            outputArray[axis] = positive - negative
        }

    }


    fun actionEvaluation() {
        actionEvaluation = true

        if (abs(valence) >= 0.0001f) backpropContribution(valence, pendingActionBucket)

        explorationSignal = false
        actionEvaluation = false
    }


    fun stateEvaluation(preState: FloatArray, postState: FloatArray) {

        if (multiplierCounter > 10) {
            adrMagMulti = adrMagAcc / multiplierCounter
            dopMagMulti = dopMagAcc / multiplierCounter
            serMagMulti = serMagAcc / multiplierCounter
            multiplierCounter = 0f
            adrMagAcc = 0f
            dopMagAcc = 0f
            serMagAcc = 0f
        }

        // ===================== Adrenaline Path =====================
        val adrMulti = if (adrMagMulti > 0) (0.2f / adrMagMulti).coerceIn(0.1f, 300f) else 1f
        val adr = evaluateSmallerBetter(preState[0], postState[0], noiseGate = 0.00001f)
        adrMagAcc += abs(adr)

        // Decreasing discomfort
        val disc = evaluateSmallerBetter(preState[24], postState[24])
        adrMagAcc += abs(disc)

        // Decreasing threat
        val thr = evaluateSmallerBetter(preState[17], postState[17])
        adrMagAcc += abs(thr)


        val adrVal = ((adr + disc + thr) * adrMulti).coerceIn(-1f, 1f)
        if (abs(adrVal) > 0f) {
            backpropContribution(adrVal, 0)
        }


        // ===================== Dopamine Path =====================
        val dopMulti = if (dopMagMulti > 0) (0.15f / dopMagMulti).coerceIn(0.1f, 200f) else 1f
        val dop = evaluateSmallerBetter(preState[1], postState[1], noiseGate = 0.0001f, inverse = true)
        dopMagAcc += abs(dop)

        // Resource distance
        val res = evaluateDecreaseDistance(preState[9], postState[9])
        dopMagAcc += abs(res)

        // Exploration pressure * hunger
        val food = evaluateSmallerBetter(preState[22], postState[22]) * postState[23]
        dopMagAcc += abs(food)

        val dopVal = ((dop + res + food) * dopMulti).coerceIn(-1f, 1f)
        if (abs(dopVal) > 0f) {
            backpropContribution(dopVal, 1)
        }


        // ===================== Serotonin Path =====================
        val serMulti = if (serMagMulti > 0) (0.15f / serMagMulti).coerceIn(0.1f, 200f) else 1f
        val ser = evaluateSmallerBetter(preState[2], postState[2], noiseGate = 0.0001f, inverse = true)
        serMagAcc += abs(ser)

        // Decreasing hunger (eating)
        val hun = evaluateSmallerBetter(preState[23], postState[23])
        serMagAcc += abs(hun)

        val serVal = ((ser + hun) * serMulti).coerceIn(-1f, 1f)
        if (abs(serVal) > 0f) {
            backpropContribution(serVal, 2)
        }

        // ===================== Locomotion Only =====================


        multiplierCounter++
    }

    var movementCounter = 0f
    var accDistanceReward = 0f
    var accAlignmentReward = 0f
    var avgDistanceReward = 0f
    var avgAlignmentReward = 0f
    fun movementEvaluation(preMotorInputs: FloatArray, postMotorInputs: FloatArray) {

        // 0 -> Alignment error
        // 1 -> Distance

        val alignmentMultiplier = if (avgAlignmentReward > 0) (0.15f / avgAlignmentReward).coerceIn(0.1f, 200f) else 1f
        val alignmentCorrection =
            evaluateSmallerBetter(preMotorInputs[0], postMotorInputs[0], noiseGate = 0f)
        val alignmentReward = alignmentCorrection * alignmentMultiplier
        accAlignmentReward += abs(alignmentReward)

        if (abs(alignmentReward) > 0) locomotionBackpropagation(alignmentReward, rotationOnly = true)

        val distanceMultiplier = if (avgDistanceReward > 0) (0.15f / avgDistanceReward).coerceIn(0.1f, 200f) else 1f
        val deltaDistance = evaluateDecreaseDistance(
            preMotorInputs[1], postMotorInputs[1], noiseGate = 0f)
        val distanceReward = deltaDistance * ((0.5f - abs(postMotorInputs[0])) * 2f) * distanceMultiplier
        accDistanceReward = abs(distanceReward)

        if (abs(distanceReward) > 0f) locomotionBackpropagation(distanceReward, rotationOnly = false)

        locomotionWeightAdjustment()

        movementCounter++

        if (movementCounter == 10f) {
            avgDistanceReward = accDistanceReward / movementCounter
            avgAlignmentReward = accAlignmentReward / movementCounter
            movementCounter = 0f
            accDistanceReward = 0f
            accAlignmentReward = 0f
        }
    }

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

    private fun evaluateDecreaseDistance(
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

    var accuReward = 0f
    var accuCounter = 0
    var avgRewMag = 0f
    fun backpropContribution(valence: Float, bucket: Int = 3) {
        appliedReward = valence

        for (i in 0 until adrenalBucket.size) adrenalBucket[i] = 0f
        for (i in 0 until adrenalBucket.size) dopaBucket[i] = 0f
        for (i in 0 until adrenalBucket.size) seroBucket[i] = 0f
        for (i in 0 until intentBucket.size) intentBucket[i] = 0f

        // Contribution calculation for pairs excluding movement axis
        for (i in 0 until outputNeurons step 2) {
            val isActionPair = outputIsAction[i]

            if (isActionPair != actionEvaluation) continue // Not updated this loop
            val outcome = outputSum[i] - outputSum[i + 1]
            val globalAxisDirection = if (isActionPair || bootstrapTimer > 0) sign(outcome) else 1f
            if (globalAxisDirection == 0f) continue

            if (isActionPair && outcome <= 0f) continue // Skip latent action pair

            val isDangerPair = i == 14 || i == 15
            val exploration = (explorationSignal || (bootstrapTimer > 0 && !actionEvaluation)) && !isDangerPair

            // Adrenal contribution
            val adrenalAxisDirection = if (exploration) 1f else sign(adrenalOutput[i] - adrenalOutput[i + 1])
            if (adrenalAxisDirection != 0f) {
                val adrenalContribution =
                    if (exploration) {
                        1f // exploratory action, calculateContribution(1) returns 0.05f otherwise - weak learning
                    } else calculateContribution(adrenalOutput[i].coerceIn(0f, 1f), skipGate = true)

                val adrenalContribution2 = calculateContribution(adrenalOutput[i + 1].coerceIn(0f, 1f), skipGate = true)
                adrenalBucket[i] += appliedReward * adrenalContribution * globalAxisDirection * adrenalAxisDirection
                adrenalBucket[i + 1] += -appliedReward * adrenalContribution2 * globalAxisDirection * adrenalAxisDirection
            }

            // Dopaminergic contribution
            val dopaminAxisDirection = if (exploration) 1f else sign(dopaminergicOutput[i] - dopaminergicOutput[i + 1])
            if (dopaminAxisDirection != 0f) {
                val dopaminergicContribution =
                    if (exploration) {
                        1f
                    } else calculateContribution(dopaminergicOutput[i].coerceIn(0f, 1f), skipGate = true)

                val dopaminergicContribution2 =
                    calculateContribution(dopaminergicOutput[i + 1].coerceIn(0f, 1f), skipGate = true)
                dopaBucket[i] += appliedReward * dopaminergicContribution * globalAxisDirection * dopaminAxisDirection
                dopaBucket[i + 1] += -appliedReward * dopaminergicContribution2 * globalAxisDirection * dopaminAxisDirection
            }

            // Serotonergic contribution
            val serotoninAxisDirection =
                if (exploration) 1f else sign(serotonergicOutput[i] - serotonergicOutput[i + 1])
            if (serotoninAxisDirection != 0f) {
                val serotonergicContribution =
                    if (exploration) {
                        1f
                    } else calculateContribution(serotonergicOutput[i].coerceIn(0f, 1f), skipGate = true)

                val serotonergicContribution2 =
                    calculateContribution(serotonergicOutput[i + 1].coerceIn(0f, 1f), skipGate = true)
                seroBucket[i] += appliedReward * serotonergicContribution * globalAxisDirection * serotoninAxisDirection
                seroBucket[i + 1] += -appliedReward * serotonergicContribution2 * globalAxisDirection * serotoninAxisDirection
            }
        }

        when (bucket) {
            0 -> {
                for (i in 0 until adrenalContribution.size) adrenalContribution[i] += adrenalBucket[i]
                for (j in 0 until intentNeurons) {
                    var totalCorrection = 0f
                    for (i in 0 until outputNeurons) {
                        totalCorrection += adrenalBucket[i] *
                                (adrenalOutputWeights[i][j] + adrenalOutputMemory[i][j])
                    }
                    intentBucket[j] += totalCorrection *
                            calculateContribution(intent[j], 0.1f)
                }
            }

            1 -> {
                for (i in 0 until dopaminergicContribution.size) dopaminergicContribution[i] += dopaBucket[i]
                for (j in 0 until intentNeurons) {
                    var totalCorrection = 0f
                    for (i in 0 until outputNeurons) {
                        totalCorrection += dopaBucket[i] *
                                (dopaminergicOutputWeights[i][j] + dopaminergicOutputMemory[i][j])
                    }
                    intentBucket[j] += totalCorrection *
                            calculateContribution(intent[j], 0.1f)
                }
            }

            2 -> {
                for (i in 0 until serotonergicContribution.size) serotonergicContribution[i] += seroBucket[i]
                for (j in 0 until intentNeurons) {
                    var totalCorrection = 0f
                    for (i in 0 until outputNeurons) {
                        totalCorrection += seroBucket[i] *
                                (serotonergicOutputWeights[i][j] + serotonergicOutputMemory[i][j])
                    }
                    intentBucket[j] += totalCorrection *
                            calculateContribution(intent[j], 0.1f)
                }
            }

            3 -> {
                for (i in 0 until adrenalContribution.size) {
                    adrenalContribution[i] += adrenalBucket[i]
                    dopaminergicContribution[i] += dopaBucket[i]
                    serotonergicContribution[i] += seroBucket[i]
                }

                val adrenalContr = adrenalBucket.sum()
                val dopaContr = dopaBucket.sum()
                val seroContr = seroBucket.sum()

                val contributionSum = (adrenalContr + dopaContr + seroContr)
                // Proportional normalization function
                val localDivisor = if (contributionSum > 1f) contributionSum else 1f


                val localAdrContr = adrenalContr / localDivisor
                val localDopaContr = dopaContr / localDivisor
                val localSeroContr = seroContr / localDivisor

                for (j in 0 until intentNeurons) {
                    var totalCorrection = 0f
                    if (abs(localAdrContr) > 0) {
                        for (i in 0 until outputNeurons) {
                            totalCorrection += adrenalBucket[i] *
                                    (adrenalOutputWeights[i][j] + adrenalOutputMemory[i][j])
                        }
                        intentBucket[j] += totalCorrection * localAdrContr *
                                calculateContribution(intent[j], 0.1f)
                    }
                    if (abs(localDopaContr) > 0) {
                        totalCorrection = 0f
                        for (i in 0 until outputNeurons) {
                            totalCorrection += dopaBucket[i] *
                                    (dopaminergicOutputWeights[i][j] + dopaminergicOutputMemory[i][j])
                        }
                        intentBucket[j] += totalCorrection * localDopaContr *
                                calculateContribution(intent[j], 0.1f)
                    }
                    if (abs(localSeroContr) > 0) {
                        totalCorrection = 0f
                        for (i in 0 until outputNeurons) {
                            totalCorrection += seroBucket[i] *
                                    (serotonergicOutputWeights[i][j] + serotonergicOutputMemory[i][j])
                        }
                        intentBucket[j] += totalCorrection * localSeroContr *
                                calculateContribution(intent[j], 0.1f)
                    }
                }


            }

        }



        for (i in 0 until intentContribution.size) intentContribution[i] += intentBucket[i]

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
        if (accuCounter > 100) {
            avgRewMag = accuReward / accuCounter
            avgValence = accuValence / accuCounter
            accuCounter = 0
            accuReward = 0f
            accuValence = 0f
        }

        this.valence = 0f
    }

    var backPropTest = 0f
    var intentTest = 0f
    var activityTest = 0f
    fun weightAdjustment() {

        // Output layer
        for (i in 0 until outputNeurons) {
            for (j in 0 until intentNeurons) {
                val adaptiveWeightDecay =
                    adaptiveWeightDecay(adrenalOutputMemory[i][j], 0.987f, 0.97f)
                val delta = adrenalContribution[i] * intent[j] * learningRate
                adrenalOutputMemory[i][j] = (adrenalOutputMemory[i][j] * adaptiveWeightDecay + delta)
                    .coerceIn(0f, maxMemory)
            }

            for (j in 0 until intentNeurons) {
                val adaptiveWeightDecay =
                    adaptiveWeightDecay(dopaminergicOutputMemory[i][j], 0.987f, 0.97f)
                val delta = dopaminergicContribution[i] * intent[j] * learningRate
                dopaminergicOutputMemory[i][j] = (dopaminergicOutputMemory[i][j] * adaptiveWeightDecay + delta)
                    .coerceIn(0f, maxMemory)
            }

            for (j in 0 until intentNeurons) {
                val adaptiveWeightDecay =
                    adaptiveWeightDecay(serotonergicOutputMemory[i][j], 0.987f, 0.97f)
                val delta = serotonergicContribution[i] * intent[j] * learningRate
                serotonergicOutputMemory[i][j] = (serotonergicOutputMemory[i][j] * adaptiveWeightDecay + delta)
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
        for (i in 0 until adrenalContribution.size) {
            backPropTest += abs(adrenalContribution[i])
            backPropTest += abs(dopaminergicContribution[i])
            backPropTest += abs(serotonergicContribution[i])
            adrenalContribution[i] = 0f
            dopaminergicContribution[i] = 0f
            serotonergicContribution[i] = 0f
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

    private fun locomotionBackpropagation(valence: Float, rotationOnly: Boolean = false) {

        for (axis in 0 until movementAxis) {
            val isRotation = axis == 0
            if (rotationOnly != isRotation) continue
            val base = axis * motorUnits * 2
            var pos = 0f;
            var neg = 0f
            for (gear in 0 until motorUnits) {
                pos += locomotionOutput[base + gear * 2]
                neg += locomotionOutput[base + gear * 2 + 1]
            }
            val globalAxisDirection = sign(pos - neg)
            if (globalAxisDirection == 0f) continue

            for (gear in 0 until motorUnits) {

                val i = base + (gear * 2)
                val contribution = calculateContribution(
                    locomotionOutput[i].coerceIn(0f, 1f), skipGate = true, eligibilityFloor = 0f
                )
                val contribution2 = calculateContribution(
                    locomotionOutput[i + 1].coerceIn(0f, 1f), skipGate = true, eligibilityFloor = 0f
                )
                motorContribution[i] += valence * contribution * globalAxisDirection
                motorContribution[i + 1] += -valence * contribution2 * globalAxisDirection

            }
        }
    }

    private fun locomotionWeightAdjustment() {
        for (i in 0 until locomotionNeurons step 2) {
            if (motorContribution[i] == 0f && motorContribution[i + 1] == 0f) continue
            val isDrive = i >= motorUnits * 2
            for (j in 0 until motionInputs) {
                val inputValue = if (isDrive && j == ROTATION_INDEX) abs(motorInputs[j]) else motorInputs[j]
                if (!isDrive && j != ROTATION_INDEX) continue
                val adaptiveWeightDecay =
                    adaptiveWeightDecay(locomotionMemory[i][j], 0.95f, 0.90f)
                val delta = inputValue * motorContribution[i] * learningRate
                locomotionMemory[i][j] = (locomotionMemory[i][j] * adaptiveWeightDecay + delta)
                    .coerceIn(-maxMemory, maxMemory)

                val adaptiveWeightDecay2 =
                    adaptiveWeightDecay(locomotionMemory[i + 1][j], 0.95f, 0.90f)
                val delta2 = inputValue * motorContribution[i + 1] * learningRate
                locomotionMemory[i + 1][j] =
                    (locomotionMemory[i + 1][j] * adaptiveWeightDecay2 + delta2)
                        .coerceIn(-maxMemory, maxMemory)
            }
        }
        for (i in 0 until motorContribution.size) motorContribution[i] = 0f
    }


    // ===================== HELPER FUNCTIONS ===================== \\

    private fun adaptiveWeightDecay(memoryValue: Float, decay: Float, maxDecay: Float): Float {
        val memory = (abs(memoryValue) / maxMemory).coerceIn(0f, 1f)
        val pressure = (memory * memory).coerceIn(0f, 1f)
        return decay - ((decay - maxDecay) * pressure)
    }

    private fun calculateContribution(
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
        val transferAdrenergic: Array<FloatArray>,
        val transferDopaminergic: Array<FloatArray>,
        val transferSerotonergic: Array<FloatArray>
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


        val exportAdrenergicOutput = Array(outputNeurons) { i ->
            adrenalOutputWeights[i].copyOf()
        }

        for (i in 0 until outputNeurons) {
            for (j in 0 until intentNeurons) {
                exportAdrenergicOutput[i][j] = (exportAdrenergicOutput[i][j] + adrenalOutputMemory[i][j]) / 2
            }
        }

        val exportDopaminergicOutput = Array(outputNeurons) { i ->
            dopaminergicOutputWeights[i].copyOf()
        }

        for (i in 0 until outputNeurons) {
            for (j in 0 until intentNeurons) {
                exportDopaminergicOutput[i][j] = (exportDopaminergicOutput[i][j] + dopaminergicOutputMemory[i][j]) / 2
            }
        }

        val exportSerotonergicOutput = Array(outputNeurons) { i ->
            serotonergicOutputWeights[i].copyOf()
        }
        for (i in 0 until outputNeurons) {
            for (j in 0 until intentNeurons) {
                exportSerotonergicOutput[i][j] = (exportSerotonergicOutput[i][j] + serotonergicOutputMemory[i][j]) / 2
            }
        }

        return WeightsPackage(
            exportInput, exportIntent,
            exportAdrenergicOutput, exportDopaminergicOutput,
            exportSerotonergicOutput
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

        val importAdrenergic = weightsPackage.transferAdrenergic
        for (i in 0 until adrenalOutputWeights.size) {
            for (j in 0 until adrenalOutputWeights[i].size) {
                adrenalOutputWeights[i][j] = importAdrenergic[i][j]
            }
        }

        val importDopaminergic = weightsPackage.transferDopaminergic
        for (i in 0 until dopaminergicOutputWeights.size) {
            for (j in 0 until dopaminergicOutputWeights[i].size) {
                dopaminergicOutputWeights[i][j] = importDopaminergic[i][j]
            }
        }

        val importSerotonergic = weightsPackage.transferSerotonergic
        for (i in 0 until serotonergicOutputWeights.size) {
            for (j in 0 until serotonergicOutputWeights[i].size) {
                serotonergicOutputWeights[i][j] = importSerotonergic[i][j]
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
        for (i in 0 until adrenalOutputMemory.size) {
            for (j in 0 until adrenalOutputMemory[i].size) {
                adrenalOutputMemory[i][j] = 0f
            }
        }
        for (i in 0 until dopaminergicOutputMemory.size) {
            for (j in 0 until dopaminergicOutputMemory[i].size) {
                dopaminergicOutputMemory[i][j] = 0f
            }
        }
        for (i in 0 until serotonergicOutputMemory.size) {
            for (j in 0 until serotonergicOutputMemory[i].size) {
                serotonergicOutputMemory[i][j] = 0f
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
                if (i >= (motorUnits * 2)) {
                    matrix[i][j] = rng.nextFloat(0f, 0.2f)
                } else {
                    matrix[i][j] = rng.nextGaussian(0.0, 0.1)
                        .toFloat().coerceIn(-0.2f, 0.2f)
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
                block.append("Adrenal [${i}][${j}]")
                block.append(" Fixed: ${adrenalOutputWeights[i][j]}")
                block.append("   Adapt: ${adrenalOutputMemory[i][j]}\n")
                block.append("Dopaminergic [${i}][${j}]")
                block.append(" Fixed: ${dopaminergicOutputWeights[i][j]}")
                block.append("   Adapt: ${dopaminergicOutputMemory[i][j]}\n")
                block.append("Serotonergic [${i}][${j}]")
                block.append(" Fixed: ${serotonergicOutputWeights[i][j]}")
                block.append("   Adapt: ${serotonergicOutputMemory[i][j]}\n")
            }
        }
        block.append("===== Output Summary: =====\n")
        for (i in 0 until outputNeurons) {
            block.append("Output[${i}]: ${outputSum[i]}\n")
        }

        block.append("===== Motor Neurons: =====\n")
        for (i in 0 until locomotionNeurons) {
            for (j in 0 until motionInputs) {
                block.append("Motor [${i}][${j}]")
                block.append(" Fixed: ${locomotionWeights[i][j]}")
                block.append("   Adapt: ${locomotionMemory[i][j]}\n")
            }
        }


        block.append("===== Motor Outputs:: =====\n")
        block.append("Rotation: ${outputs[0]}\n")
        block.append("Drive: ${outputs[1]}\n")

        block.append("===== Network Outputs:: =====\n")

        block.append("Kill: ${outputs[2]}\n")
        block.append("Mate: ${outputs[3]}\n")
        block.append("Eat: ${outputs[4]}\n")
        block.append("Target Intent: ${outputs[5]}\n")
        block.append("Food Intent: ${outputs[6]}\n")
        block.append("Familiar Intent: ${outputs[7]}\n")
        block.append("Unfamiliar Intent: ${outputs[8]}\n")
        block.append("Danger Intent: ${outputs[9]}\n")

        block.append("===== Accessories: =====\n")
        block.append("AVG REWARD (100 ticks): $avgValence \n")
        block.append("AVG REWARD MAG (100 ticks): $avgRewMag \n")
        block.append("ADR REWARD MAG: $adrMagMulti \n")
        block.append("DOP REWARD MAG: $dopMagMulti \n")
        block.append("SER REWARD MAG: $serMagMulti \n")
        block.append("Avg. Movement Reward: $avgDistanceReward \n")
        //block.append("IntentMag: ${intent.sumOf  { abs(it).toDouble() } / intent.size} \n")
        //block.append("ActivityMag: ${activity.sumOf { abs(it).toDouble() } / activity.size } \n")
        block.append("===== Is currently: =====\n")
        if ((outputSum[0] - outputSum[1]) >= 0.5f && !explorationSignal) block.append("Attempting to Kill\n")
        if ((outputSum[2] - outputSum[3]) >= 0.5f && !explorationSignal) block.append("Attempting to Mate\n")
        if ((outputSum[4] - outputSum[5]) >= 0.5f && !explorationSignal) block.append("Attempting to Eat\n")
        if (bootstrapTimer > 0) {
            when (bootstrappedIntent) {
                1 -> block.append("Learning Target\n")
                2 -> block.append("Learning Food\n")
                3 -> block.append("Learning Familiar\n")
                4 -> block.append("Learning Unfamiliar\n")
            }
        }
        when (COMMITED_INTENT) {
            0 -> block.append("Exploring\n")
            1 -> if ((outputSum[6] - outputSum[7]) > 0)
                block.append("Chasing Target\n") else block.append("Fleeing from Target\n")
            2 -> if ((outputSum[8] - outputSum[9]) > 0)
                block.append("Chasing Food\n") else block.append("Fleeing from Food\n")
            3 -> if ((outputSum[10] - outputSum[11]) > 0)
                block.append("Chasing Familiar\n") else block.append("Fleeing from Familiar\n")
            4 -> if ((outputSum[12] - outputSum[13]) > 0)
                block.append("Chasing Unfamiliar\n") else block.append("Fleeing from Unfamiliar\n")
        }
        if ((outputSum[14] - outputSum[15]) > 0)
            block.append("Approaching Danger\n") else block.append("Fleeing from Danger\n")
        block.append("===== Accessories: =====\n")
        block.append("===== END =====\n")
        return block.toString()
    }

}

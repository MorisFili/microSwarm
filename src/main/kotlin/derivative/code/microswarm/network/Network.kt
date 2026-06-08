package derivative.code.microswarm.network

import java.util.*
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sign
import kotlin.math.tanh


class Network(
    val inputStates: Int,
    val movementAxis: Int,
    val actionPairs: Int,
    val preferenceNeurons: Int
) {

    // Fixed Variables
    val networkOutputs = movementAxis + actionPairs + preferenceNeurons
    private val inputNeurons = inputStates
    private val activityNeurons = inputStates
    private val intentNeurons = inputStates / 2
    private val outputNeurons = ((movementAxis + actionPairs) * 2) + preferenceNeurons
    private val learningRate = 0.08f
    private val weightDecay = 0.99985f // For the deepest layer
    private val maxMemory = 1.0f

    //private val activationThreshold = 0.13f
    val ACTION_THRESHOLD = 0.5f

    // Util
    private val rng = Random()

    // Genetic weights
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
    private val hiddenBiasWeights: FloatArray = FloatArray(activityNeurons)

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
    private var input = FloatArray(inputNeurons)
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
    var metaData = FloatArray(10) // Unused for now
    var valence = 0f
    var actionEvaluation = false
    var explorationSignal = false
    var pendingActionBucket = 3

    // Avg reward debug
    var adrMagMulti = 1f
    var dopMagMulti = 1f
    var serMagMulti = 1f
    var adrMagAcc = 0f
    var dopMagAcc = 0f
    var serMagAcc = 0f
    var multiplierCounter = 0f


    init {
        arrayBiasSeeder(hiddenBiasWeights)
        matrixBiasSeeder(weightsInput)
        matrixBiasSeeder(weightsIntent)
        matrixBiasSeeder(adrenalOutputWeights)
        matrixBiasSeeder(dopaminergicOutputWeights)
        matrixBiasSeeder(serotonergicOutputWeights)
        val movementIndices = movementAxis * 2
        val actionIndices = actionPairs * 2
        for (i in movementIndices until movementIndices + actionIndices) { // Output indices that are action based
            outputIsAction[i] = true
        }
        seedAllReflexes()
    }


    fun feedForward(inputArray: FloatArray, outputArray: FloatArray) {

        for (i in 0 until inputNeurons) input[i] = inputArray[i] // Snapshot

        for (i in 0 until activityNeurons) {
            var sum = hiddenBiasWeights[i]
            for (j in 0 until inputNeurons) {
                val weight = weightsInput[i][j] + memoryIn[i][j]
                sum += weight * input[j]
            }
            // Leaky ReLU
            activity[i] = (if (sum > 0f) sum else sum * 0.25f).coerceIn(-0.25f, 1f)
        }

        for (i in 0 until intentNeurons) {
            var sum = 0f
            for (j in 0 until activityNeurons) {
                val weight = weightsIntent[i][j] + memoryIntent[i][j]
                sum += weight * activity[j]
            }
            intent[i] = tanh(sum)
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
        val KILL = (outputSum[4] - outputSum[5]).coerceIn(0.05f, 1f)
        val MATE = (outputSum[6] - outputSum[7]).coerceIn(0.05f, 1f)
        val EAT = (outputSum[8] - outputSum[9]).coerceIn(0.05f, 1f)

        val CHOICE = maxOf(KILL, MATE, EAT)

        if (CHOICE < ACTION_THRESHOLD) {

            val ACTION_CHANCE = CHOICE / ACTION_THRESHOLD

            if (rng.nextFloat() < ACTION_CHANCE) {
                explorationSignal = true
                val CHANCE = (KILL + MATE + EAT) * rng.nextFloat()
                if (CHANCE < KILL) {
                    outputSum[4] = 1f
                    outputSum[5] = 0f
                    outputSum[6] = 0f
                    outputSum[7] = 0f
                    outputSum[8] = 0f
                    outputSum[9] = 0f
                    pendingActionBucket = 0
                } else if (CHANCE < KILL + MATE) {
                    outputSum[4] = 0f
                    outputSum[5] = 0f
                    outputSum[6] = 1f
                    outputSum[7] = 0f
                    outputSum[8] = 0f
                    outputSum[9] = 0f
                    pendingActionBucket = 1
                } else {
                    outputSum[4] = 0f
                    outputSum[5] = 0f
                    outputSum[6] = 0f
                    outputSum[7] = 0f
                    outputSum[8] = 1f
                    outputSum[9] = 0f
                    pendingActionBucket = 2
                }
            }

        } else if (CHOICE >= ACTION_THRESHOLD) {
            // Keep winner, flatten rest
            when (CHOICE) {
                KILL -> {
                    outputSum[6] = 0f
                    outputSum[7] = 0f
                    outputSum[8] = 0f
                    outputSum[9] = 0f
                    pendingActionBucket = 0
                }

                MATE -> {
                    outputSum[4] = 0f
                    outputSum[5] = 0f
                    outputSum[8] = 0f
                    outputSum[9] = 0f
                    pendingActionBucket = 1
                }

                EAT -> {
                    outputSum[4] = 0f
                    outputSum[5] = 0f
                    outputSum[6] = 0f
                    outputSum[7] = 0f
                    pendingActionBucket = 2
                }
            }

        }


        val actionPairsIndex = movementAxis + actionPairs
        for (i in 0 until actionPairsIndex) outputArray[i] = outputSum[i * 2] - outputSum[i * 2 + 1]
        for (i in actionPairsIndex until networkOutputs) outputArray[i] = outputSum[i + actionPairsIndex]

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
        val disc = evaluateSmallerBetter(preState[35], postState[35])
        adrMagAcc += abs(disc)

        // Decreasing threat
        val thr = evaluateSmallerBetter(preState[28], postState[28])
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
        val res = evaluateDecreaseDistance(preState[17], postState[17])
        dopMagAcc += abs(res)

        // Food exploration
        val food = evaluateSmallerBetter(preState[33], postState[33]) * postState[34]
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
        val hun = evaluateSmallerBetter(preState[34], postState[34])
        serMagAcc += abs(hun)

        val serVal = ((ser + hun) * serMulti).coerceIn(-1f, 1f)
        if (abs(serVal) > 0f) {
            backpropContribution(serVal, 2)
        }

        // ===================== Ambiguous Path =====================
        // Target distance
        val tar = evaluateDecreaseDistance(preState[20], postState[20])
        backpropContribution(tar)


        multiplierCounter++
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
        inverse: Boolean = false
    ): Float {
        if (preState == 0f || postState == 0f) return 0f
        val delta = (abs(preState) - abs(postState))
        if (abs(delta) < 0.001f) return 0f
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

        // Contribution calculation for pairs
        for (i in 0 until outputNeurons - preferenceNeurons step 2) {
            val isActionPair = outputIsAction[i]

            if (isActionPair != actionEvaluation) continue // Not updated this loop
            val outcome = outputSum[i] - outputSum[i + 1]
            val globalAxisDirection = sign(outcome)
            if (globalAxisDirection == 0f) continue

            if (isActionPair && outcome <= 0f) continue // Skip latent action pair

            // Adrenal contribution
            val adrenalAxisDirection = if (explorationSignal) 1f else sign(adrenalOutput[i] - adrenalOutput[i + 1])
            val adrenalContribution =
                if (explorationSignal) {
                    1f // exploratory action, calculateContribution(1) returns 0.05f otherwise - weak learning
                } else calculateContribution(adrenalOutput[i].coerceIn(0f, 1f), skipGate = true)

            val adrenalContribution2 = calculateContribution(adrenalOutput[i + 1].coerceIn(0f, 1f), skipGate = true)
            adrenalBucket[i] += appliedReward * adrenalContribution * globalAxisDirection * adrenalAxisDirection
            adrenalBucket[i + 1] += -appliedReward * adrenalContribution2 * globalAxisDirection * adrenalAxisDirection

            // Dopaminergic contribution
            val dopaminAxisDirection = if (explorationSignal) 1f else sign(dopaminergicOutput[i] - dopaminergicOutput[i + 1])
            val dopaminergicContribution =
                if (explorationSignal) {
                    1f
                } else calculateContribution(dopaminergicOutput[i].coerceIn(0f, 1f), skipGate = true)

            val dopaminergicContribution2 =
                calculateContribution(dopaminergicOutput[i + 1].coerceIn(0f, 1f), skipGate = true)
            dopaBucket[i] += appliedReward * dopaminergicContribution * globalAxisDirection * dopaminAxisDirection
            dopaBucket[i + 1] += -appliedReward * dopaminergicContribution2 * globalAxisDirection * dopaminAxisDirection

            // Serotonergic contribution
            val serotoninAxisDirection = if (explorationSignal) 1f else sign(serotonergicOutput[i] - serotonergicOutput[i + 1])
            val serotonergicContribution =
                if (explorationSignal) {
                    1f
                } else calculateContribution(serotonergicOutput[i].coerceIn(0f, 1f), skipGate = true)

            val serotonergicContribution2 =
                calculateContribution(serotonergicOutput[i + 1].coerceIn(0f, 1f), skipGate = true)
            seroBucket[i] += appliedReward * serotonergicContribution * globalAxisDirection * serotoninAxisDirection
            seroBucket[i + 1] += -appliedReward * serotonergicContribution2 * globalAxisDirection * serotoninAxisDirection

        }

        // Contribution calculation for singles (currently only target preferences)
        for (i in outputNeurons - preferenceNeurons until outputNeurons) {
            if (!actionEvaluation) continue // Evaluate only during action evaluation (targeting)

            val adrenalContribution = calculateContribution(adrenalOutput[i].coerceIn(-1f, 1f))
            adrenalBucket[i] += appliedReward * adrenalContribution

            val dopaContribution = calculateContribution(dopaminergicOutput[i].coerceIn(-1f, 1f))
            dopaBucket[i] += appliedReward * dopaContribution

            val seroContribution = calculateContribution(serotonergicOutput[i].coerceIn(-1f, 1f))
            seroBucket[i] += appliedReward * seroContribution
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
        if (accuCounter > 100) {
            avgRewMag = accuReward / accuCounter
            accuCounter = 0
            accuReward = 0f
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
                    .coerceIn(-maxMemory, maxMemory)
            }

            for (j in 0 until intentNeurons) {
                val adaptiveWeightDecay =
                    adaptiveWeightDecay(dopaminergicOutputMemory[i][j], 0.987f, 0.97f)
                val delta = dopaminergicContribution[i] * intent[j] * learningRate
                dopaminergicOutputMemory[i][j] = (dopaminergicOutputMemory[i][j] * adaptiveWeightDecay + delta)
                    .coerceIn(-maxMemory, maxMemory)
            }

            for (j in 0 until intentNeurons) {
                val adaptiveWeightDecay =
                    adaptiveWeightDecay(serotonergicOutputMemory[i][j], 0.987f, 0.97f)
                val delta = serotonergicContribution[i] * intent[j] * learningRate
                serotonergicOutputMemory[i][j] = (serotonergicOutputMemory[i][j] * adaptiveWeightDecay + delta)
                    .coerceIn(-maxMemory, maxMemory)
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
                val delta = input[k] * activityContribution[j] * learningRate
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


        resetMutables()
        seedAllReflexes()
    }

    private fun seedAllReflexes() {

    }

    private fun resetMutables() {
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

    private fun matrixBiasSeeder(matrix: Array<FloatArray>) {
        for (i in 0 until matrix.size) {
            for (j in 0 until matrix[i].size) {
                matrix[i][j] = rng.nextGaussian(0.0, 0.1)
                    .toFloat().coerceIn(-0.2f, 0.2f)
            }
        }
    }

    fun analysis(inputs: FloatArray): String {
        val block = StringBuilder(5000)
        block.append("===== START =====\n")
        for (i in inputs.indices) {
            block.append("Input(${i}): ${inputs[i]}\n")
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
        block.append("===== Action: =====\n")

        block.append("X-axis: ${outputSum[0] - outputSum[1]}\n")
        block.append("Y-axis: ${outputSum[2] - outputSum[3]}\n")
        block.append("Kill: ${outputSum[4] - outputSum[5]}\n")
        block.append("Mate: ${outputSum[6] - outputSum[7]}\n")
        block.append("Eat: ${outputSum[8] - outputSum[9]}\n")


        block.append("===== Accessories: =====\n")
        block.append("AVG REWARD MAG (100 ticks): $avgRewMag \n")
        block.append("ADR REWARD MAG: $adrMagMulti \n")
        block.append("DOP REWARD MAG: $dopMagMulti \n")
        block.append("SER REWARD MAG: $serMagMulti \n")
        block.append("===== END =====\n")
        return block.toString()
    }

}

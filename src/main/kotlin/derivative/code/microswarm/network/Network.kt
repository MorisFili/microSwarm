package derivative.code.microswarm.network

import derivative.code.microswarm.discountLookup
import java.util.*
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.sign
import kotlin.math.tanh


class Network(
    val networkInput: Int,
    val networkOutput: Int
) {

    // Fixed Variables
    private val inputNeurons = networkInput
    private val activityNeurons = networkInput
    private val intentNeurons = networkInput / 2
    private val outputNeurons = networkOutput * 2
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
    private val intentInputWeights: Array<FloatArray> =
        Array(intentNeurons) { FloatArray(inputNeurons) }
    private val outputIntentWeights: Array<FloatArray> =
        Array(outputNeurons) { FloatArray(intentNeurons) }
    private val hiddenBiasWeights: FloatArray = FloatArray(activityNeurons)

    // Plastic weights
    private val memoryIn: Array<FloatArray> =
        Array(activityNeurons) { FloatArray(inputNeurons) }
    private val memoryIntent: Array<FloatArray> =
        Array(intentNeurons) { FloatArray(activityNeurons) }
    private val intentInputMemory: Array<FloatArray> =
        Array(intentNeurons) { FloatArray(inputNeurons) }
    private val outputIntentMemory: Array<FloatArray> =
        Array(outputNeurons) { FloatArray(intentNeurons) }
    private val outputIsAction = BooleanArray(outputNeurons) { false }

    // Rolling Variables
    private var input = FloatArray(inputNeurons)
    private var intent = FloatArray(intentNeurons)
    private var interactionIntent = FloatArray(intentNeurons)
    private var activity = FloatArray(activityNeurons)
    private var output = FloatArray(outputNeurons)
    private val backpropContribution = FloatArray(outputNeurons)
    private val backpropBucket = FloatArray(outputNeurons)
    private val intentContribution = FloatArray(intentNeurons)
    private val intentBucket = FloatArray(intentNeurons)
    private val activityContribution = FloatArray(activityNeurons)
    private var appliedReward = 0f
    var metaData = FloatArray(10) // Arbitrary size for now
    var valence = 0f
    var actionReward = false
    var explorationSignal = false

    // Avg reward debug
    var rewCounter = 0
    var accuReward = 0f
    var avgReward = 0f
    var contributingStatesCounter = 0

    init {
        arrayBiasSeeder(hiddenBiasWeights)
        matrixBiasSeeder(weightsInput)
        matrixBiasSeeder(weightsIntent)
        matrixBiasSeeder(intentInputWeights)
        matrixBiasSeeder(outputIntentWeights)
        for (i in 4..9) { // Output indices that are action based
            outputIsAction[i] = true
        }
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

            var inputContext = 0f
            for (k in 0 until inputNeurons) {
                inputContext += input[k] * (intentInputWeights[i][k] + intentInputMemory[i][k])
            }
            interactionIntent[i] = tanh(inputContext)
        }

        for (i in 0 until outputNeurons) {
            var sumOI = 0f
            var sumInt = 0f
            for (j in 0 until intentNeurons) {
                val weightsOI = outputIntentWeights[i][j] + outputIntentMemory[i][j]
                sumOI += weightsOI * intent[j]
                sumInt += weightsOI * interactionIntent[j]
            }
            val interaction = sumOI * sumInt
            val drive = if (abs(interaction) > abs(sumOI)) interaction else sumOI
            output[i] = max(0f, drive) // ReLU
        }

        // Softmax-Argmax-like logic for actions
        val KILL = (output[4] - output[5]).coerceIn(0.05f, 1f)
        val MATE = (output[6] - output[7]).coerceIn(0.05f, 1f)
        val WORK = (output[8] - output[9]).coerceIn(0.05f, 1f)

        if (maxOf(KILL, MATE, WORK) < ACTION_THRESHOLD) {

            val ACTION_CHANCE = maxOf(KILL, MATE, WORK) / ACTION_THRESHOLD

            if (rng.nextFloat() < ACTION_CHANCE) {
                explorationSignal = true
                val CHANCE = (KILL + MATE + WORK) * rng.nextFloat()
                if (CHANCE < KILL) {
                    output[4] = 1f
                    output[5] = 0f
                    output[6] = 0f
                    output[7] = 0f
                    output[8] = 0f
                    output[9] = 0f
                } else if (CHANCE < KILL + MATE) {
                    output[4] = 0f
                    output[5] = 0f
                    output[6] = 1f
                    output[7] = 0f
                    output[8] = 0f
                    output[9] = 0f
                } else {
                    output[4] = 0f
                    output[5] = 0f
                    output[6] = 0f
                    output[7] = 0f
                    output[8] = 1f
                    output[9] = 0f
                }
            }

        } else if (maxOf(KILL, MATE, WORK) >= ACTION_THRESHOLD) {
            // Keep winner, flatten rest
            if (KILL >= MATE && KILL >= WORK) {
                output[6] = 0f
                output[7] = 0f
                output[8] = 0f
                output[9] = 0f
            } else if (MATE >= KILL && MATE >= WORK) {
                output[4] = 0f
                output[5] = 0f
                output[8] = 0f
                output[9] = 0f
            } else {
                output[4] = 0f
                output[5] = 0f
                output[6] = 0f
                output[7] = 0f
            }
        }

        // Random exploration when standing still
        val moveX = output[0] - output[1]
        val moveY = output[2] - output[3]

        if (abs(moveX) < 0.001f && abs(moveY) < 0.001f && rng.nextFloat() < 0.2f) {
            output[0] = rng.nextFloat(1f)
            output[1] = rng.nextFloat(1f)
            output[2] = rng.nextFloat(1f)
            output[3] = rng.nextFloat(1f)
        }

        for (i in 0 until networkOutput) outputArray[i] = output[i * 2] - output[i * 2 + 1]


    }

    fun actionEvaluation() {
        actionReward = true

        // Currency accumulation
        valence += evaluateResourceGain(metaData[0], metaData[1], metaData[2])

        if (abs(valence) >= 0.0001f) backpropContribution(valence)

        explorationSignal = false
        actionReward = false
    }


    fun stateEvaluation(preState: FloatArray, postState: FloatArray) {

        val detectionRadius = 40f
        val targetRadius = detectionRadius / 2f

        // sameHue attraction DX + DY * hueStrength
        valence += evaluateSmallerBetter(
            preState[1], postState[1],
            detectionRadius, postState[4]
        )
        valence += evaluateSmallerBetter(
            preState[2], postState[2],
            detectionRadius, postState[4]
        )

        // otherHue avoidance DX + DY * hueStrength
        valence += evaluateSmallerBetter(
            preState[5], postState[5],
            detectionRadius, postState[8], negative = true
        )
        valence += evaluateSmallerBetter(
            preState[6], postState[6],
            detectionRadius, postState[8], negative = true
        )

        // Renegade avoidance DX + DY * renegadeStrength
        valence += evaluateSmallerBetter(
            preState[11], postState[11],
            detectionRadius, postState[14], negative = true
        )
        valence += evaluateSmallerBetter(
            preState[12], postState[12],
            detectionRadius, postState[14], negative = true
        )

        // Crowding
        valence += evaluateSmallerBetter(
            preState[18], postState[18], // Possibly too weak
            1f
        ).coerceIn(0f, 1f)
        // Alone
        valence += evaluateSmallerBetter(
            preState[19], postState[19], // Possibly too weak
            1f
        ).coerceIn(0f, 1f)

        // Target chasing distanceDelta * 1-distance (distance remaining scalar)
        valence += evaluateSmallerBetter(
            preState[23], postState[23],
            targetRadius, (1 - postState[23])
        )

        // Resource center distance
        valence += evaluateSmallerBetter(preState[27], postState[27], detectionRadius)

        valence /= max(1, contributingStatesCounter)

        if (abs(valence) > 0.001f) {
            val adaptiveMultiplier = (0.2f / avgReward.coerceAtLeast(0.0001f)).coerceIn(0.1f, 10f)
            val finalValence = valence * adaptiveMultiplier
            backpropContribution(finalValence.coerceIn(-1f, 1f))
        } else valence = 0f
        contributingStatesCounter = 0
    }

    private fun evaluateSmallerBetter(
        preState: Float,
        postState: Float,
        maxRadius: Float,
        scalar: Float = 1f,
        negative: Boolean = false
    ): Float {
        if (preState == 0f || postState == 0f) return 0f // skip inactive
        val delta = abs(preState) - abs(postState)
        if (abs(delta) < 0.000001f) return 0f
        val perfectDelta = 1 / maxRadius
        val perfectStepsToTarget = ceil(abs(postState) * maxRadius)
        val discountBonus = 1 - discountLookup[perfectStepsToTarget.toInt().coerceIn(0, discountLookup.lastIndex)]
        val reward = (delta / perfectDelta).coerceIn(-1f, 1f)

        val valenceOut = (abs(reward) + (discountBonus * 0.5f)) *
                sign(reward) * scalar *
                if (negative) -1f else 1f // sign flip for negative

        if (abs(valenceOut) > 0.1f) contributingStatesCounter++ // Significant enough contribution

        return valenceOut
    }

    private fun evaluateResourceGain(preValue: Float, postValue: Float, divisor: Float): Float {
        if (preValue <= 0f) return 0f
        val normalizedPre = preValue / divisor
        val normalizedPost = postValue / divisor
        val relativeGain = normalizedPost / normalizedPre
        return (relativeGain - 1f).coerceIn(-1f, 1f)
    }

    fun backpropContribution(valence: Float) {
        appliedReward = valence

        for (i in 0 until backpropBucket.size) backpropBucket[i] = 0f
        for (i in 0 until intentBucket.size) intentBucket[i] = 0f

        // Contribution calculation
        for (i in 0 until outputNeurons step 2) {
            val isActionPair = outputIsAction[i]

            if (isActionPair != actionReward) continue // Not updated this loop
            val outcome = output[i] - output[i + 1]
            val axisDirection = sign(outcome)
            if (axisDirection == 0f) continue

            if (isActionPair && outcome <= 0f) continue // Skip latent action pair

            val contributionGate =
                if (explorationSignal) {
                    1f // exploratory action, calculateContribution(1) returns 0.05f otherwise - weak learning
                } else calculateContribution(output[i].coerceIn(0f, 1f), skipGate = true)


            val contributionGate2 = calculateContribution(output[i + 1].coerceIn(0f, 1f), skipGate = true)



            backpropBucket[i] += appliedReward * contributionGate * axisDirection
            backpropBucket[i + 1] += -appliedReward * contributionGate2 * axisDirection

        }

        for (i in 0 until backpropContribution.size) backpropContribution[i] += backpropBucket[i]

        for (j in 0 until intentNeurons) {
            var totalCorrection = 0f
            for (i in 0 until outputNeurons) {
                totalCorrection += backpropBucket[i] *
                        (outputIntentWeights[i][j] + outputIntentMemory[i][j])
            }
            intentBucket[j] += totalCorrection *
                    calculateContribution(intent[j], 0.1f)
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

        rewCounter++
        accuReward += abs(valence)
        if (rewCounter > 10) {
            avgReward = accuReward / rewCounter
            rewCounter = 0
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
                    adaptiveWeightDecay(outputIntentMemory[i][j], 0.987f, 0.97f)
                val delta = backpropContribution[i] * intent[j] * learningRate
                outputIntentMemory[i][j] = (outputIntentMemory[i][j] * adaptiveWeightDecay + delta)
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

        for (j in 0 until intentNeurons) {
            for (k in 0 until inputNeurons) {
                val adaptiveWeightDecay =
                    adaptiveWeightDecay(intentInputMemory[j][k], 0.9975f, 0.994f)
                val delta = input[k] * intentContribution[j] * learningRate
                intentInputMemory[j][k] = (intentInputMemory[j][k] * adaptiveWeightDecay + delta)
                    .coerceIn(-maxMemory, maxMemory)
            }
        }
        backPropTest = 0f
        intentTest = 0f
        activityTest = 0f
        // Post adjustment cleanup
        for (i in 0 until backpropContribution.size) {
            backPropTest += abs(backpropContribution[i])
            backpropContribution[i] = 0f
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
        val transferIntentInput: Array<FloatArray>,
        val transferOutputIntent: Array<FloatArray>
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

        val exportIntentInput = Array(intentNeurons) { i ->
            intentInputWeights[i].copyOf()
        }
        for (i in 0 until intentNeurons) {
            for (j in 0 until inputNeurons) {
                exportIntentInput[i][j] = (exportIntentInput[i][j] + intentInputMemory[i][j]) / 2
            }
        }

        val exportOutputIntent = Array(outputNeurons) { i ->
            outputIntentWeights[i].copyOf()
        }

        for (i in 0 until outputNeurons) {
            for (j in 0 until intentNeurons) {
                exportOutputIntent[i][j] = (exportOutputIntent[i][j] + outputIntentMemory[i][j]) / 2
            }
        }

        return WeightsPackage(
            exportInput, exportIntent,
            exportIntentInput, exportOutputIntent
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

        val importIntentInput = weightsPackage.transferIntentInput
        for (i in 0 until intentInputWeights.size) {
            for (j in 0 until intentInputWeights[i].size) {
                intentInputWeights[i][j] = importIntentInput[i][j]
            }
        }

        val importOutputIntent = weightsPackage.transferOutputIntent
        for (i in 0 until outputIntentWeights.size) {
            for (j in 0 until outputIntentWeights[i].size) {
                outputIntentWeights[i][j] = importOutputIntent[i][j]
            }
        }
        resetMutables()
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
        for (i in 0 until intentInputMemory.size) {
            for (j in 0 until intentInputMemory[i].size) {
                intentInputMemory[i][j] = 0f
            }
        }
        for (i in 0 until outputIntentMemory.size) {
            for (j in 0 until outputIntentMemory[i].size) {
                outputIntentMemory[i][j] = 0f
            }
        }

        // Rolling variables
        valence = 0f
        actionReward = false
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

    private fun matrixBiasSeeder3d(matrix: Array<Array<FloatArray>>) {
        for (i in 0 until matrix.size) {
            for (j in 0 until matrix[i].size) {
                for (k in 0 until matrix[i][j].size) {
                    matrix[i][j][k] = rng.nextGaussian(0.0, 0.1)
                        .toFloat().coerceIn(-0.2f, 0.2f)
                }
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
        block.append("===== Intent Input Neurons: =====\n")
        for (i in 0 until intentNeurons) {
            for (j in 0 until inputNeurons) {
                block.append("IntentInput[${i}][${j}]")
                block.append(" Fixed: ${intentInputWeights[i][j]}")
                block.append("   Adapt: ${intentInputMemory[i][j]}\n")
            }
            block.append("Intent[${i}]: ${interactionIntent[i]}\n")
        }
        block.append("===== OutputIntent Neurons: =====\n")
        for (i in 0 until outputNeurons) {
            for (j in 0 until intentNeurons) {
                block.append("OutputIntent [${i}][${j}]")
                block.append(" Fixed: ${outputIntentWeights[i][j]}")
                block.append("   Adapt: ${outputIntentMemory[i][j]}\n")
            }
        }
        block.append("===== Output Neurons: =====\n")
        for (i in 0 until outputNeurons) {
            block.append("Output[${i}]: ${output[i]}\n")
        }
        block.append("===== Action: =====\n")
        for (i in 0 until networkOutput) {
            block.append("Action[${i}]: ${output[i * 2] - output[i * 2 + 1]}\n")
        }

        block.append("===== Accessories: =====\n")
        block.append("EFFECTIVE REWARD: $appliedReward \n")
        block.append("AVG REWARD MAG: $avgReward \n")
        block.append("===== END =====\n")
        return block.toString()
    }

}

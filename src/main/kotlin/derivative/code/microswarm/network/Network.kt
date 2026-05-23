package derivative.code.microswarm.network

import derivative.code.microswarm.Main
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
    private val learningRate = 0.08f + (0.08f * Main.LR_AMP)
    private val weightDecay = 0.995f
    private val maxMemory = 1.5f
    //private val activationThreshold = 0.13f
    val ACTION_THRESHOLD = 0.5f

    // Util
    private val rng = Random()

    // Genetic weights
    private val weightsInput: Array<FloatArray> =
        Array(activityNeurons) { FloatArray(inputNeurons) }
    private val weightsIntent: Array<FloatArray> =
        Array(intentNeurons) { FloatArray(activityNeurons) }
    private val outputIntentWeights: Array<FloatArray> =
        Array(outputNeurons) { FloatArray(intentNeurons) }
    private val outputIntentInputWeights =
        Array(outputNeurons) { Array(intentNeurons) { FloatArray(inputNeurons) } }
    private val hiddenBiasWeights: FloatArray = FloatArray(activityNeurons)

    // Plastic weights
    private val memoryIn: Array<FloatArray> =
        Array(activityNeurons) { FloatArray(inputNeurons) }
    private val memoryIntent: Array<FloatArray> =
        Array(intentNeurons) { FloatArray(activityNeurons) }
    private val outputIntentMemory: Array<FloatArray> =
        Array(outputNeurons) { FloatArray(intentNeurons) }
    private val outputIntentInputMemory =
        Array(outputNeurons) { Array(intentNeurons) { FloatArray(inputNeurons) } }
    private val outputIsAction = BooleanArray(outputNeurons) { false }

    // Rolling Variables
    private var input = FloatArray(inputNeurons)
    private var intent = FloatArray(intentNeurons)
    private var activity = FloatArray(activityNeurons)
    private var output = FloatArray(outputNeurons)
    private var appliedReward = 0f
    var valence = 0f
    var actionReward = false
    var intentWithoutAction = false


    init {
        arrayBiasSeeder(hiddenBiasWeights)
        matrixBiasSeeder(weightsInput)
        matrixBiasSeeder(weightsIntent)
        matrixBiasSeeder(outputIntentWeights)
        matrixBiasSeeder3d(outputIntentInputWeights)
        for (i in 4..9) { // Output indices that are action based
            outputIsAction[i] = true
        }
    }


    fun feedForward(input: FloatArray): FloatArray {
        this.input = input.copyOf() // Snapshot

        for (i in 0 until activityNeurons) {
            var sum = hiddenBiasWeights[i]
            for (j in 0 until inputNeurons) {
                val weight = weightsInput[i][j] + memoryIn[i][j]
                sum += weight * input[j]
            }
            // Leaky ReLU
            activity[i] = (if (sum > 0f) sum else sum * 0.25f).coerceIn(-0.25f, 1f)
        }

        var totalAbs = 0f
        for (i in 0 until intentNeurons) {
            var sum = 0f
            for (j in 0 until activityNeurons) {
                val weight = weightsIntent[i][j] + memoryIntent[i][j]
                sum += weight * activity[j]
            }
            intent[i] = tanh(sum).coerceIn(-0.9f, 0.9f)
            intent[i] = intent[i] / max(0.05f, 1f - intent[i] * intent[i]) // Hyperbolic inflation
            totalAbs += abs(intent[i])
        }
        if (totalAbs > 1f) { // Normalization
            for (i in 0 until intentNeurons) {
                intent[i] = intent[i] / totalAbs
            }
        }

        for (i in 0 until outputNeurons) {

            var sumOI = 0f
            for (j in 0 until intentNeurons) {
                val weight = outputIntentWeights[i][j] + outputIntentMemory[i][j]
                sumOI += weight * intent[j]
            }

            var sumOII = 0f
            for (l in 0 until intentNeurons) {
                for (m in 0 until inputNeurons) {
                    val weight = outputIntentInputWeights[i][l][m] + outputIntentInputMemory[i][l][m]
                    sumOII += weight * intent[l] * input[m]
                }
            }

            output[i] = max(0f, sumOI + sumOII) // ReLU
        }

        // Softmax-Argmax-like logic for actions
        val KILL = (output[4] - output[5]).coerceIn(0.05f, 1f)
        val MATE = (output[6] - output[7]).coerceIn(0.05f, 1f)
        val WORK = (output[8] - output[9]).coerceIn(0.05f, 1f)

        if (maxOf(KILL, MATE, WORK) < ACTION_THRESHOLD) {

            val ACTION_CHANCE = maxOf(KILL, MATE, WORK) / ACTION_THRESHOLD

            if (rng.nextFloat() < ACTION_CHANCE) {
                intentWithoutAction = true
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

        if (moveX < 0.001f && moveY < 0.001f && rng.nextFloat() < 0.2f) {
            output[0] = rng.nextFloat(0.5f)
            output[1] = rng.nextFloat(0.5f)
            output[2] = rng.nextFloat(0.5f)
            output[3] = rng.nextFloat(0.5f)
        }

        return FloatArray(networkOutput) { i ->
            output[i * 2] - output[i * 2 + 1]
        }
    }

    fun actionEvaluation() {
        if (abs(valence) < 0.0001f) return
        actionReward = true
        outputFeedback(valence)
        intentWithoutAction = false
        actionReward = false
    }

    fun stateEvaluation(preState: FloatArray, postState: FloatArray) {

        valence += evaluateSmallerBetter(preState[1], postState[1]) * postState[4]
        valence += evaluateSmallerBetter(preState[2], postState[2]) * postState[4]

        valence += evaluateSmallerBetter(preState[5], postState[5], negative = true) * postState[8]
        valence += evaluateSmallerBetter(preState[6], postState[6], negative = true) * postState[8]

        valence += evaluateResourceGain(preState[10], postState[10])

        valence += evaluateSmallerBetter(preState[11], postState[11], negative = true) * postState[14]
        valence += evaluateSmallerBetter(preState[12], postState[12], negative = true) * postState[14]

        valence += evaluateSmallerBetter(preState[18], postState[18])

        outputFeedback(valence)
    }

    private fun evaluateSmallerBetter(preState: Float, postState: Float, negative: Boolean = false): Float {
        val delta = abs(preState) - abs(postState)
        if (abs(delta) < 0.000001f) return 0f

        val stepsToTarget = ceil(abs(postState) / abs(delta))
        val expectedReward = 1 - discountLookup[stepsToTarget.toInt().coerceIn(0, discountLookup.lastIndex)]

        return ((abs(delta) * 5f) +
                (expectedReward * 0.5f)) *
                sign(delta) *
                if (negative) -1f else 1f // sign flip for negative
    }

    private fun evaluateResourceGain(preState: Float, postState: Float, negative: Boolean = false): Float {
        val relativeGain = if (preState > 0 ) postState / preState else 1f
        return (relativeGain - 1f).coerceIn(-1f, 1f)
    }

    fun outputFeedback(reward: Float) {
        val adaptiveWeightDecay = if (abs(reward) > 0.05f) weightDecay else 0.99995f
        appliedReward = if (abs(reward) < 0.05f && abs(reward) > 0.001) 0.05f * sign(reward) else reward


        // Contribution calculation
        val outputContribution = FloatArray(outputNeurons)

        for (i in 0 until outputNeurons step 2) {
            val isActionOutput = outputIsAction[i]

            if (isActionOutput != actionReward) continue // Not updated this loop
            if (isActionOutput && output[i] <= 0.0001f) continue // Skip inactive action neuron

            val contributionScale =
                if (intentWithoutAction) 1f
                else maxOf(0.05f, 1f - output[i] * output[i])

            val contributionScale2 = maxOf(0.05f, 1f - output[i + 1] * output[i + 1])

            val pairContributionScale = 0.5f

            if (isActionOutput) {
                outputContribution[i] = appliedReward * contributionScale * pairContributionScale
                outputContribution[i + 1] = -appliedReward * contributionScale * pairContributionScale
            } else {
                outputContribution[i] = appliedReward * contributionScale
                outputContribution[i + 1] = appliedReward * contributionScale2
            }
        }

        val intentContribution = FloatArray(intentNeurons)
        for (j in 0 until intentNeurons) {
            var totalCorrection = 0f

            for (i in 0 until outputNeurons) {
                totalCorrection += outputContribution[i] *
                        (outputIntentWeights[i][j] + outputIntentMemory[i][j])

                for (k in 0 until inputNeurons) {
                    totalCorrection += outputContribution[i] *
                            (outputIntentInputWeights[i][j][k] + outputIntentInputMemory[i][j][k]) *
                            input[k]
                }
            }
            val intentDPost = maxOf(0.05f, 1f - intent[j] * intent[j])
            intentContribution[j] = totalCorrection * intentDPost
        }

        val activityContribution = FloatArray(activityNeurons)
        for (j in 0 until activityNeurons) {
            val activityDPost = if (activity[j] > 0) 1f else 0.25f
            var totalCorrection = 0f
            for (i in 0 until intentNeurons) {
                totalCorrection += intentContribution[i] * (weightsIntent[i][j] + memoryIntent[i][j])
            }

            activityContribution[j] = totalCorrection * activityDPost
        }


        // Output layer
        for (i in 0 until outputNeurons) {
            if (outputIsAction[i] != actionReward) continue

            for (j in 0 until intentNeurons) {
                val delta = outputContribution[i] * intent[j] * learningRate
                outputIntentMemory[i][j] = (outputIntentMemory[i][j] * adaptiveWeightDecay + delta)
                    .coerceIn(-maxMemory, maxMemory)
            }

            for (j in 0 until intentNeurons) {
                for (k in 0 until inputNeurons) {
                    val delta = outputContribution[i] * intent[j] * input[k] * learningRate
                    outputIntentInputMemory[i][j][k] = (outputIntentInputMemory[i][j][k] * adaptiveWeightDecay + delta)
                        .coerceIn(-maxMemory, maxMemory)
                }
            }
        }

        // Hidden Layer
        // Update runs twice per tick now and this layer isn't be gated by outputIsAction[i]
        // Minimal fix = reduce magnitude by half
        val sharedBackpropScale = 0.5f
        for (i in 0 until intentNeurons) {
            for (j in 0 until activityNeurons) {
                val delta = intentContribution[i] * activity[j] *
                        learningRate * sharedBackpropScale
                memoryIntent[i][j] = (memoryIntent[i][j] * adaptiveWeightDecay + delta)
                    .coerceIn(-maxMemory, maxMemory)
            }
        }

        for (j in 0 until activityNeurons) {
            for (k in 0 until inputNeurons) {
                val delta = input[k] * activityContribution[j] *
                        learningRate * sharedBackpropScale
                memoryIn[j][k] = (memoryIn[j][k] * adaptiveWeightDecay + delta)
                    .coerceIn(-maxMemory, maxMemory)
            }
        }

        valence = 0f
    }


    // ===================== HELPER FUNCTIONS ===================== \\

    class WeightsPackage(
        val transferInput: Array<FloatArray>,
        val transferIntent: Array<FloatArray>,
        val transferOutputIntent: Array<FloatArray>,
        val transferOutputIntentInput: Array<Array<FloatArray>>
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

        val exportOutputIntent = Array(outputNeurons) { i ->
            outputIntentWeights[i].copyOf()
        }

        val exportOutputIntentInput = Array(outputNeurons) { i ->
            Array(intentNeurons) { l ->
                outputIntentInputWeights[i][l].copyOf()
            }
        }

        for (i in 0 until outputNeurons) {
            for (j in 0 until intentNeurons) {
                exportOutputIntent[i][j] = (exportOutputIntent[i][j] + outputIntentMemory[i][j]) / 2
            }

            for (l in 0 until intentNeurons) {
                for (m in 0 until inputNeurons) {
                    exportOutputIntentInput[i][l][m] =
                        (exportOutputIntentInput[i][l][m] + outputIntentInputMemory[i][l][m]) / 2
                }
            }
        }

        return WeightsPackage(exportInput, exportIntent, exportOutputIntent, exportOutputIntentInput)
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

        val importOutputIntent = weightsPackage.transferOutputIntent
        for (i in 0 until outputIntentWeights.size) {
            for (j in 0 until outputIntentWeights[i].size) {
                outputIntentWeights[i][j] = importOutputIntent[i][j]
            }
        }

        val importOutputIntentInput = weightsPackage.transferOutputIntentInput
        for (i in 0 until outputIntentInputWeights.size) {
            for (j in 0 until outputIntentInputWeights[i].size) {
                for (k in 0 until outputIntentInputWeights[i][j].size) {
                    outputIntentInputWeights[i][j][k] = importOutputIntentInput[i][j][k]
                }
            }
        }
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
        block.append("===== Memory weights: =====\n\n")
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
        block.append("===== OutputIntent Neurons: =====\n")
        for (i in 0 until outputNeurons) {
            for (j in 0 until intentNeurons) {
                block.append("OutputIntent [${i}][${j}]")
                block.append(" Fixed: ${outputIntentWeights[i][j]}")
                block.append("   Adapt: ${outputIntentMemory[i][j]}\n")
            }
        }

        block.append("===== OutputIntentInput Neurons: =====\n")
        for (i in 0 until outputNeurons) {
            for (j in 0 until intentNeurons) {
                for (k in 0 until inputNeurons) {
                    block.append("OutputIntentInput [${i}][${j}][${k}]")
                    block.append(" Fixed: ${outputIntentInputWeights[i][j][k]}")
                    block.append("   Adapt: ${outputIntentInputMemory[i][j][k]}\n")
                }
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
        block.append("===== END =====\n")
        return block.toString()
    }

}
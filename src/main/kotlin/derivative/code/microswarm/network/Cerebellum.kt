package derivative.code.microswarm.network

import derivative.code.microswarm.Simulation.Companion.rng
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sign

object Cerebellum {

    fun generateMovement(self: Cortex, motorInputArray: FloatArray, outputArray: FloatArray) {
        for (i in 0 until self.motionInputs) self.motorInputs[i] = motorInputArray[i]

        // Locomotion block
        for (i in 0 until self.locomotionNeurons) {
            var sum = 0f
            val isDrive = i >= (self.motorUnits * 2) // First half of locomotion neurons is rotation, second half is drive
            for (j in 0 until self.motionInputs) {
                if (!isDrive && j != self.ROTATION_INDEX) continue // Distance shouldn't affect rotation
                val weight = self.locomotionWeights[i][j] + self.locomotionMemory[i][j]
                val inputs = if (isDrive && j == self.ROTATION_INDEX)
                    abs(self.motorInputs[j]) else self.motorInputs[j]
                sum += weight * inputs
            }
            val torque = (((i / 2) % 8) * self.sharePerUnit) + self.sharePerUnit + 1f
            self.locomotionOutput[i] = max(0f, sum * torque).coerceAtMost(1f)
        }


        for (axis in 0 until self.movementAxis) {
            val base = axis * self.motorUnits * 2
            var positive = 0f
            var negative = 0f
            for (gear in 0 until self.motorUnits) {
                positive += self.locomotionOutput[base + (gear * 2)]
                negative += self.locomotionOutput[base + (gear * 2) + 1]
            }
            if (axis != self.ROTATION_INDEX && positive < 0.001f && negative < 0.001f && rng.nextFloat() < 0.1f) {
                self.locomotionOutput[base] = rng.nextFloat() * self.sharePerUnit
                self.locomotionOutput[base + 1] = rng.nextFloat() * self.sharePerUnit
                positive = self.locomotionOutput[base]
                negative = self.locomotionOutput[base + 1]
            }

            outputArray[axis] = positive - negative
        }
    }

    fun movementEvaluation(self: Cortex, preMotorInputs: FloatArray, postMotorInputs: FloatArray, steerCorrection: Float) {

        // 0 -> Alignment error
        // 1 -> Distance

        val alignmentMultiplier = if (self.avgAlignmentReward > 0) (0.15f / self.avgAlignmentReward).coerceIn(0.1f, 200f) else 1f
        val alignmentCorrection =
            self.evaluateRotation(preMotorInputs[0], steerCorrection)
        val alignmentReward = alignmentCorrection * alignmentMultiplier
        self.accAlignmentReward += abs(alignmentReward)

        if (abs(alignmentReward) > 0) locomotionBackpropagation(self, alignmentReward, rotationOnly = true)

        val distanceMultiplier = if (self.avgDistanceReward > 0) (0.15f / self.avgDistanceReward).coerceIn(0.1f, 200f) else 1f
        val deltaDistance = self.evaluateDecreaseDistance(
            preMotorInputs[1], postMotorInputs[1], noiseGate = 0f
        )
        val distanceReward = deltaDistance * distanceMultiplier
        self.accDistanceReward += abs(distanceReward)

        if (abs(distanceReward) > 0f) locomotionBackpropagation(self, distanceReward, rotationOnly = false)

        locomotionWeightAdjustment(self)

        self.movementCounter++

        if (self.movementCounter == 10f) {
            self.avgDistanceReward = self.accDistanceReward / self.movementCounter
            self.avgAlignmentReward = self.accAlignmentReward / self.movementCounter
            self.movementCounter = 0f
            self.accDistanceReward = 0f
            self.accAlignmentReward = 0f
        }
    }

    private fun locomotionBackpropagation(self: Cortex, valence: Float, rotationOnly: Boolean = false) {

        for (axis in 0 until self.movementAxis) {
            val isRotation = axis == 0
            if (rotationOnly != isRotation) continue // Rotation & Drive separate backprop
            val base = axis * self.motorUnits * 2
            var pos = 0f;
            var neg = 0f
            for (gear in 0 until self.motorUnits) {
                pos += self.locomotionOutput[base + gear * 2]
                neg += self.locomotionOutput[base + gear * 2 + 1]
            }
            val globalAxisDirection = sign(pos - neg)
            if (globalAxisDirection == 0f) continue

            for (gear in 0 until self.motorUnits) {

                val i = base + (gear * 2)
                val contribution = self.calculateContribution(
                    self.locomotionOutput[i].coerceIn(0f, 1f), skipGate = true, eligibilityFloor = 0f
                )
                val contribution2 = self.calculateContribution(
                    self.locomotionOutput[i + 1].coerceIn(0f, 1f), skipGate = true, eligibilityFloor = 0f
                )
                self.motorContribution[i] += valence * contribution * globalAxisDirection
                self.motorContribution[i + 1] += -valence * contribution2 * globalAxisDirection

            }
        }
    }

    private fun locomotionWeightAdjustment(self: Cortex) {
        for (i in 0 until self.locomotionNeurons step 2) {
            if (self.motorContribution[i] == 0f && self.motorContribution[i + 1] == 0f) continue
            val isDrive = i >= self.motorUnits * 2
            for (j in 0 until self.motionInputs) {
                if (!isDrive && j != self.ROTATION_INDEX) continue
                val adaptiveWeightDecay =
                    self.adaptiveWeightDecay(self.locomotionMemory[i][j], 0.9975f, 0.99f)
                val delta = self.motorInputs[j] * self.motorContribution[i] * self.locomotionLearningRate
                self.locomotionMemory[i][j] = (self.locomotionMemory[i][j] * adaptiveWeightDecay + delta)
                    .coerceIn(-self.maxMemory, self.maxMemory)

                val adaptiveWeightDecay2 =
                    self.adaptiveWeightDecay(self.locomotionMemory[i + 1][j], 0.99f, 0.99f)
                val delta2 = self.motorInputs[j] * self.motorContribution[i + 1] * self.locomotionLearningRate
                self.locomotionMemory[i + 1][j] =
                    (self.locomotionMemory[i + 1][j] * adaptiveWeightDecay2 + delta2)
                        .coerceIn(-self.maxMemory, self.maxMemory)
            }
        }
        for (i in 0 until self.motorContribution.size) self.motorContribution[i] = 0f
    }

}
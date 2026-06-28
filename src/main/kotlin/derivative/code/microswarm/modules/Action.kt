package derivative.code.microswarm.modules

import derivative.code.microswarm.INV_MAX_ENERGY
import derivative.code.microswarm.Main
import derivative.code.microswarm.Simulation
import derivative.code.microswarm.TABLE_SIZE
import derivative.code.microswarm.cosTable
import derivative.code.microswarm.entity.Agent
import derivative.code.microswarm.entity.GeneticMaterial
import derivative.code.microswarm.entity.State
import derivative.code.microswarm.entity.Target
import derivative.code.microswarm.managePopHueCounter
import derivative.code.microswarm.network.Cortex
import derivative.code.microswarm.network.CortexIndex
import derivative.code.microswarm.network.Intents
import derivative.code.microswarm.sinTable
import kotlin.math.abs
import kotlin.math.sign

object Action {

    object Perform {
        const val ACTION_ENERGY_TRANSFER_AMT = 10f


        fun performAction(self: Agent) {

            val network = self.networkAccess()

            val KILL = 1
            val MATE = 2
            val EAT = 3
            val SHARE = 4
            val STEAL = 5

            val choice = network.committedActionIntent

            if (choice == 0) return

            when (choice) {
                EAT -> {
                    val food = self.foodField ?: return
                    if (!self.digesting) {
                        food.value--
                        self.digesting = true
                        self.digestionTimer = 10
                    }

                }

                else -> {
                    if (self.TARGET == null) return
                    val target = self.TARGET as Agent
                    if (!target.enabled) return

                    if (!self.withinActionRadius) return

                    when (choice) {
                        MATE -> mate(self, target)
                        KILL -> kill(self, target)
                        SHARE -> share(self, target)
                        STEAL -> steal(self, target)

                    }


                }
            }
        }

        fun move(self: Agent, rotate: Float, drive: Float) {
            val movement = drive.coerceIn(-1f, 1f)
            val rotation = rotate.coerceIn(-1f, 1f)
            self.movementMetaData[0] = rotation

            for (i in self.networkAccess().foodMemoryRot.indices) {
                self.networkAccess().foodMemoryRot[i] -= rotation
            }

            val angleIndex = (((rotation + 1) / 2) * TABLE_SIZE).toInt().coerceIn(0, TABLE_SIZE - 1)
            val cosA = cosTable[angleIndex]
            val sinA = sinTable[angleIndex]

            val newFX = self.facingX * cosA - self.facingY * sinA
            val newFY = self.facingX * sinA + self.facingY * cosA

            self.facingX = newFX
            self.facingY = newFY

            var newX = self.x + (self.facingX * movement)
            var newY = self.y + (self.facingY * movement)

            if (newX >= 999.99f) newX -= 999.99f
            if (newX < 0f) newX += 999.99f
            if (newY >= 999.99f) newY -= 999.99f
            if (newY < 0f) newY += 999.99f

            // Occupancy check
            val gridX = newX.toInt()
            val gridY = newY.toInt()
            val oldGridX = self.x.toInt()
            val oldGridY = self.y.toInt()
            Simulation.occupancyGrid[oldGridX][oldGridY] = false // clear old occupancy

            if (Simulation.occupancyGrid[gridX][gridY]) { // If new position is occupied
                // Try X
                if (!Simulation.occupancyGrid[gridX][oldGridY]) {
                    self.x = newX
                }
                // Try Y
                if (!Simulation.occupancyGrid[oldGridX][gridY]) {
                    self.y = newY
                }
            } else {
                self.x = newX
                self.y = newY
            }
            Simulation.occupancyGrid[self.x.toInt()][self.y.toInt()] = true
        }


        private fun kill(self: Agent, target: Agent) {
            // Death related logic
            if (!target.enabled) return
            target.enabled = false

            // Witnessed a killing
            var witnessValence = 0f
            for (bystander in self.agentsInProximity) {
                if (bystander == null) continue
                if (!bystander.enabled) continue
                val bystanderToTargetValence =
                    Target.getTargetValence(bystander, target.id).coerceIn(-1f, 1f)
                Target.upsertTargetValence(bystander, self.id, -bystanderToTargetValence)
                Target.removeTargetFromMemory(bystander, target.id)
                witnessValence -= bystanderToTargetValence
            }
            self.globalPopularity += witnessValence
            Target.removeTargetFromMemory(self, target.id)
            self.targetCooldown = 50

            Target.clearTarget(self)
            if (target.hue == self.hue) Main.Companion.killedOwnHue.incrementAndGet()
            else Main.killedOtherHue.incrementAndGet()

            managePopHueCounter(target.hue, false)
            Main.populationCounter.decrementAndGet()
            Simulation.occupancyGrid[target.x.toInt()][target.y.toInt()] = false // clear occupancy
        }

        private fun mate(self: Agent, target: Agent) {

            val sameHue = if (target.hue == self.hue) 1f else -1f
            val oppositeSex = if (target.isMale != self.isMale) 1f else -1f
            val myValenceAtTarget = Target.getTargetValence(target, self.id).coerceIn(-1f, 1f)
            val isIncubating = if (self.INCUBATING) -1f else 1f
            val myStanding = (self.localPopularity).coerceAtMost(1f)
            val myHunger = 1 - (self.energy * INV_MAX_ENERGY)

            if (self.isMale != target.isMale) {

                val targetApproval = target.networkAccess().experimentalTargetAssessment(
                    sameHue, oppositeSex, myValenceAtTarget, isIncubating,
                    myStanding, myHunger
                )

                if (targetApproval < 0.2f) { // Shunned
                    Target.upsertTargetValence(self, target.id, -0.1f)
                    Target.upsertTargetValence(target, self.id, -0.1f)
                    self.targetCooldown = 10
                    self.globalPopularity -= 0.1f
                    return
                }


                if (self.isMale) { // is male
                    if (!target.INCUBATING) {

                        target.INCUBATING = true
                        self.energy -= 20
                        if (self.hue == target.hue) Main.Companion.spawnedWithOwnHue.incrementAndGet()
                        else Main.Companion.spawnedWithOtherHue.incrementAndGet()
                        target.INCUBATION_TIMER = self.INCUBATION_LENGTH
                        target.INCUBATION_MATERIAL =
                            GeneticMaterial(self.networkAccess().extractWeightsForReproduction(), self.hue!!)

                        Target.upsertTargetValence(self, target.id, 1f)
                        Target.upsertTargetValence(target, self.id, 1f)
                        target.globalPopularity += 1f
                        self.globalPopularity += 1f
                    }

                } else if (!self.INCUBATING) {
                    self.INCUBATING = true
                    target.energy -= 20
                    if (self.hue == target.hue) Main.spawnedWithOwnHue.incrementAndGet()
                    else Main.spawnedWithOtherHue.incrementAndGet()
                    self.INCUBATION_TIMER = self.INCUBATION_LENGTH
                    self.INCUBATION_MATERIAL =
                        GeneticMaterial(
                            target.networkAccess().extractWeightsForReproduction(),
                            self.hue!!
                        )
                    Target.upsertTargetValence(self, target.id, 1f)
                    Target.upsertTargetValence(target, self.id, 1f)
                    target.globalPopularity += 1f
                    self.globalPopularity += 1f
                }

                Target.upsertTargetValence(self, target.id, 0.5f)
                Target.upsertTargetValence(target, self.id, 0.5f)
                target.globalPopularity += 0.5f
                self.globalPopularity += 0.5f

                self.targetCooldown = 50

                return

            }

        }

        private fun share(self: Agent, target: Agent) {
            target.energy += ACTION_ENERGY_TRANSFER_AMT
            self.energy -= ACTION_ENERGY_TRANSFER_AMT

            val significance = (ACTION_ENERGY_TRANSFER_AMT / target.energy.coerceAtLeast(0f)).coerceIn(0f, 1f)

            // Witnesses
            var witnessValence = 0f
            for (bystander in self.agentsInProximity) {
                if (bystander == null) continue
                if (!bystander.enabled) continue
                val concern = Target.getTargetValence(bystander, target.id).coerceIn(-1f, 1f) *
                        significance
                Target.upsertTargetValence(bystander, self.id, concern)
                witnessValence += concern
            }

            Target.upsertTargetValence(target, self.id, significance)
            self.globalPopularity += witnessValence

            self.targetCooldown = 10

            if (target.hue == self.hue) Main.sharedWithSameHue.incrementAndGet()
            else Main.sharedWithOtherHue.incrementAndGet()
        }

        private fun steal(self: Agent, target: Agent) {
            target.energy -= ACTION_ENERGY_TRANSFER_AMT
            self.energy += ACTION_ENERGY_TRANSFER_AMT

            if (target.energy <= 0) {
                kill(self, target)
                if (target.hue == self.hue) Main.killedOwnHueByStealing.incrementAndGet()
                else Main.killedOtherHueByStealing.incrementAndGet()
                return
            }

            val significance = (ACTION_ENERGY_TRANSFER_AMT / target.energy.coerceAtLeast(0f)).coerceIn(0f, 1f)

            // Witnesses
            var witnessValence = 0f
            for (bystander in self.agentsInProximity) {
                if (bystander == null) continue
                if (!bystander.enabled) continue
                val concern = Target.getTargetValence(bystander, target.id).coerceIn(0f, 1f) * significance
                Target.upsertTargetValence(bystander, self.id, -concern)
                witnessValence -= concern
            }

            Target.upsertTargetValence(target, self.id, -significance)

            self.globalPopularity += witnessValence
            self.targetCooldown = 10

            if (target.hue == self.hue) Main.Companion.stoleFromSameHue.incrementAndGet()
            else Main.Companion.stoleFromOtherHue.incrementAndGet()
        }

    }

    object Network {
        private val NONE = intArrayOf(-1)
        private val EAT_BIT = 1
        private val KILL_BIT = 2
        private val STEAL_BIT = 4
        private val SHARE_BIT = 8
        private val MATE_BIT = 16

        private val TABLE: Array<IntArray> = Array(32) { mask ->
            val list = mutableListOf<Int>()
            if (mask and EAT_BIT != 0) list.add(Intents.EAT)
            if (mask and KILL_BIT != 0) list.add(Intents.KILL)
            if (mask and STEAL_BIT != 0) list.add(Intents.STEAL)
            if (mask and SHARE_BIT != 0) list.add(Intents.SHARE)
            if (mask and MATE_BIT != 0) list.add(Intents.MATE)
            if (list.isEmpty()) NONE else list.toIntArray()
        }

        private fun getPossibleExploration(eat: Boolean, kill: Boolean, steal: Boolean, share: Boolean, mate: Boolean): IntArray {
            var mask = 0
            if (eat) mask += EAT_BIT
            if (kill) mask += KILL_BIT
            if (steal) mask += STEAL_BIT
            if (share) mask += SHARE_BIT
            if (mate) mask += MATE_BIT
            return TABLE[mask]
        }


        private fun exploration(self: Cortex): IntArray {
            // ================== UNDER CONSTRUCTION =====================
            // 0,1 -> KILL
            // 2,3 -> MATE
            // 4,5 -> EAT
            // 6,7 -> SHARE
            // 8,9 -> STEAL
            // EAT and MATE is the only "save" non-consequential exploration decision, rest
            // needs to be gated behind reflex-like circumstances

            val withinEatingRange = self.stateInputs[State.Index.IN_FOOD_FIELD] == 1f
            val withinActionRadius = self.stateInputs[State.Index.WITHIN_ACTION_RADIUS] == 1f

            val hungerWithNoFood = self.stateInputs[State.Index.HUNGER] > 0.7f && self.stateInputs[State.Index.RESOURCE_DIST] == 0f
            val threatFromTarget = self.stateInputs[State.Index.TARGET_THREAT] > 0.5f

            val wellFed = self.stateInputs[State.Index.HUNGER] < 0.3f

            return getPossibleExploration(
                withinEatingRange,
                withinActionRadius && threatFromTarget,
                hungerWithNoFood && withinActionRadius,
                wellFed && withinActionRadius,
                wellFed && withinActionRadius
            )
        }

        fun actionEvaluation(self: Cortex, neurotransmitters: FloatArray) {
            self.actionEvaluation = true

            if (self.committedActionIntent == 0) return // No action committed

            val prediction = self.output[CortexIndex.A_PREDICTION]

            val error = ((neurotransmitters[1] - self.serotonin) - (neurotransmitters[0] - self.adrenaline)) - prediction



            if (abs(error) > self.actionWeightGate) {
                self.predictionSignalUpdate(error, CortexIndex.A_PREDICTION)
                actionWeightUpdates(self, error)
                self.actionWeightGate = abs(error)
            }
            self.actionWeightGate -= 0.001f
            self.actionEvaluation = false
        }

        fun generateAction(self: Cortex) {
            // Softmax-Argmax-like logic for actions
            // 0,1 -> KILL
            // 2,3 -> MATE
            // 4,5 -> EAT
            // 6,7 -> SHARE
            // 8,9 -> STEAL

            val explorationChoice = exploration(self).random()

            val explorationChoiceMag = when (explorationChoice) {
                1 -> self.output[CortexIndex.A_KILL] + self.output[CortexIndex.A_KILL + 1]
                2 -> self.output[CortexIndex.A_MATE] + self.output[CortexIndex.A_MATE + 1]
                3 -> self.output[CortexIndex.A_EAT] + self.output[CortexIndex.A_EAT + 1]
                4 -> self.output[CortexIndex.A_SHARE] + self.output[CortexIndex.A_SHARE + 1]
                5 -> self.output[CortexIndex.A_STEAL] + self.output[CortexIndex.A_STEAL + 1]
                else -> 1f
            }

            if (explorationChoiceMag < self.bootStrengthGate) {

                when (explorationChoice) { // Exploratory signal injection
                    1 -> {
                        self.output[CortexIndex.A_KILL] = 0.5f
                        self.output[CortexIndex.A_KILL + 1] = 0f
                    }

                    2 -> {
                        self.output[CortexIndex.A_MATE] = 0.5f
                        self.output[CortexIndex.A_MATE + 1] = 0f
                    }

                    3 -> {
                        self.output[CortexIndex.A_EAT] = 0.5f
                        self.output[CortexIndex.A_EAT + 1] = 0f
                    }

                    4 -> {
                        self.output[CortexIndex.A_SHARE] = 0.5f
                        self.output[CortexIndex.A_SHARE + 1] = 0f
                    }

                    5 -> {
                        self.output[CortexIndex.A_STEAL] = 0.5f
                        self.output[CortexIndex.A_STEAL + 1] = 0f
                    }
                }
            }

            val KILL =
                (self.output[CortexIndex.A_KILL] - self.output[CortexIndex.A_KILL + 1]).coerceIn(0.01f, 1f) *
                        self.actionHabituation[1] * self.spatialIntentSalienceFactor(Intents.KILL)
            val MATE =
                (self.output[CortexIndex.A_MATE] - self.output[CortexIndex.A_MATE + 1]).coerceIn(0.01f, 1f) *
                        self.actionHabituation[2] * self.spatialIntentSalienceFactor(Intents.MATE)
            val EAT =
                (self.output[CortexIndex.A_EAT] - self.output[CortexIndex.A_EAT + 1]).coerceIn(0.01f, 1f) *
                        self.actionHabituation[3] * self.spatialIntentSalienceFactor(Intents.EAT)
            val SHARE =
                (self.output[CortexIndex.A_SHARE] - self.output[CortexIndex.A_SHARE + 1]).coerceIn(0.01f, 1f) *
                        self.actionHabituation[4] * self.spatialIntentSalienceFactor(Intents.SHARE)
            val STEAL =
                (self.output[CortexIndex.A_STEAL] - self.output[CortexIndex.A_STEAL + 1]).coerceIn(0.01f, 1f) *
                        self.actionHabituation[5] * self.spatialIntentSalienceFactor(Intents.STEAL)

            val CHOICE = maxOf(KILL, MATE, EAT, SHARE, STEAL)

            if (CHOICE >= self.COMMITMENT_GATE) {
                // Keep winner, flatten rest
                when (CHOICE) {
                    KILL -> {
                        self.committedActionIntent = 1
                    }

                    MATE -> {
                        self.committedActionIntent = 2
                    }

                    EAT -> {
                        self.committedActionIntent = 3
                    }

                    SHARE -> {
                        self.committedActionIntent = 4
                    }

                    STEAL -> {
                        self.committedActionIntent = 5
                    }
                }

            } else self.committedActionIntent = 0


            for (a in 1..5) {
                val survivalBypass = a == 3 && self.stateInputs[State.Index.HUNGER] > 0.4f
                self.actionHabituation[a] = when {
                    survivalBypass -> 1f
                    a == self.committedActionIntent -> (self.actionHabituation[a] * 0.996f).coerceAtLeast(0.3f)
                    else -> (self.actionHabituation[a] + 0.002f).coerceAtMost(1f)
                }
            }
        }


        fun actionWeightUpdates(self: Cortex, error: Float) {
            // Contribution calculation for pairs excluding movement axis
            for (i in CortexIndex.ACTION_INDEX_START..CortexIndex.ACTION_INDEX_END step 2) {

                if (self.eligibilityGate(i)) continue

                val axisDirection = sign(self.output[i] - self.output[i + 1])
                if (axisDirection != 0f) {
                    val contribution = self.calculateContribution(self.output[i].coerceIn(0f, 1f), skipGate = true)
                    val contribution2 = self.calculateContribution(self.output[i + 1].coerceIn(0f, 1f), skipGate = true)
                    self.actionOutputContribution[i] += contribution * axisDirection  * error
                    self.actionOutputContribution[i + 1] += -contribution2 * axisDirection  * error
                }
            }


            for (j in 0 until self.intentNeurons) {
                val intentDPost = if (self.intent[j] > 0) 1f else 0.25f
                var totalCorrection = 0f
                for (i in CortexIndex.ACTION_INDEX_START..CortexIndex.ACTION_INDEX_END) {
                    totalCorrection += self.actionOutputContribution[i] * (self.outputWeights[i][j] + self.outputMemory[i][j])
                }
                self.actionIntentContribution[j] += totalCorrection * intentDPost *
                        self.calculateContribution(self.intent[j], 0.1f)
            }


            for (j in 0 until self.activityNeurons) {
                val activityDPost = if (self.activity[j] > 0) 1f else 0.25f
                var totalCorrection = 0f
                for (i in 0 until self.intentNeurons) {
                    totalCorrection += self.actionIntentContribution[i] * (self.weightsIntent[i][j] + self.memoryIntent[i][j])
                }
                self.actionActivityContribution[j] += totalCorrection * activityDPost *
                        self.calculateContribution(self.activity[j], 0.1f)
            }


            for (i in CortexIndex.ACTION_INDEX_START..CortexIndex.ACTION_INDEX_END) {
                if (self.eligibilityGate(i)) continue
                for (j in 0 until self.intentNeurons) {
                    val adaptiveWeightDecay =
                        self.adaptiveWeightDecay(self.outputMemory[i][j],
                            self.outputLayerDecayBase, self.outputLayerDecayMax)
                    val delta = self.actionOutputContribution[i] * self.intent[j] * self.learningMultiplier
                    self.outputMemory[i][j] = (self.outputMemory[i][j] * adaptiveWeightDecay + delta)
                        .coerceIn(0f, self.maxMemory)
                    if (abs(self.outputMemory[i][j]) > 0f)
                        self.outputWeights[i][j] = (self.outputWeights[i][j] * self.convergenceRate) +
                                ((1 - self.convergenceRate) * self.outputMemory[i][j])
                }
            }

            for (i in 0 until self.intentNeurons) {
                for (j in 0 until self.activityNeurons) {
                    val adaptiveWeightDecay =
                        self.adaptiveWeightDecay(self.memoryIntent[i][j],
                            self.intentLayerDecayBase, self.intentLayerDecayMax)
                    val delta = self.actionIntentContribution[i] * self.activity[j] * self.learningMultiplier
                    self.memoryIntent[i][j] = (self.memoryIntent[i][j] * adaptiveWeightDecay + delta)
                        .coerceIn(-self.maxMemory, self.maxMemory)
                    if (abs(self.memoryIntent[i][j]) > 0f)
                        self.weightsIntent[i][j] = (self.weightsIntent[i][j] * self.convergenceRate) +
                                ((1 - self.convergenceRate) * self.memoryIntent[i][j])

                }
            }

            for (j in 0 until self.activityNeurons) {
                for (k in 0 until self.inputNeurons) {
                    val adaptiveWeightDecay = self.weightDecay // 0.99985
                    val delta = self.actionActivityContribution[j] * self.stateInputs[k] * self.learningMultiplier
                    self.memoryIn[j][k] = (self.memoryIn[j][k] * adaptiveWeightDecay + delta)
                        .coerceIn(-self.maxMemory, self.maxMemory)
                    if (abs(self.memoryIn[j][k]) > 0f)
                        self.weightsInput[j][k] = (self.weightsInput[j][k] * self.convergenceRate) +
                                ((1 - self.convergenceRate) * self.memoryIn[j][k])

                }
            }

            self.learningMultiplier = 1f


            for (i in 0 until self.actionOutputContribution.size) self.actionOutputContribution[i] = 0f
            for (i in 0 until self.actionIntentContribution.size) self.actionIntentContribution[i] = 0f
            for (i in 0 until self.actionActivityContribution.size) self.actionActivityContribution[i] = 0f
        }


    }
}
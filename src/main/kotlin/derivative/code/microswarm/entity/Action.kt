package derivative.code.microswarm.entity

import derivative.code.microswarm.GROUP_SIZE
import derivative.code.microswarm.INV_MAX_ENERGY
import derivative.code.microswarm.MAX_ENERGY
import derivative.code.microswarm.Main
import derivative.code.microswarm.Simulation
import derivative.code.microswarm.TABLE_SIZE
import derivative.code.microswarm.cosTable
import derivative.code.microswarm.managePopHueCounter
import derivative.code.microswarm.sinTable
import kotlin.math.max

object Action {

    val baseStealCost = 0.25f
    val witnessStealCost = 0.1f

    fun performAction(self: Agent, output: FloatArray) {

        val KILL = output[2]
        val MATE = output[3]
        val EAT = output[4]
        val SHARE = output[5]
        val STEAL = output[6]

        val network = self.networkAccess()

        val outcomeSimulation = network.explorationSignal

        val choice = maxOf(KILL, MATE, EAT, SHARE, STEAL)

        if (choice < self.networkAccess().COMMITMENT_GATE) return

        if (outcomeSimulation) {
            when (choice) {
                EAT -> {
                    // Teaching eat is good
                    val energyRatio = self.ENERGY / MAX_ENERGY
                    network.valence += if (energyRatio <= 1f) {
                        1f
                    } else {
                        -(energyRatio - 1f) * 0.3f  // slight punishment for overeating, scaled down
                    }
                    return
                }

                else -> {

                    if (self.TARGET == null) return
                    if (!self.withinActionRadius) return
                    val target = self.TARGET as Agent
                    if (!target.enabled) return

                    val targetSocialValence =
                        if (target.hue == self.hue) Target.getTargetValence(self, target.id) + 0.1f
                        else Target.getTargetValence(self, target.id)
                    val ratio = (self.ENERGY * 0.01f)

                    when (choice) {
                        MATE -> {
                            network.valence += self.preState[1] * self.targetAttractiveness
                            return
                        }

                        KILL -> {
                            val enmity = max(0f, -targetSocialValence) // Hostility towards target
                            val threat = self.threatFromTarget * (1 - self.targetDistance)
                            val targetFriendsCount = target.friendsInProximityCount
                            val targetWitnessCount = target.agentsInProximityCount
                            val targetPopularity = target.localPopularity

                            val nonFriends = targetWitnessCount - targetFriendsCount
                            val enemiesEstimate = nonFriends * ((1f - targetPopularity) / 2f).coerceIn(0f, 1f)
                            val neutralsEstimate = nonFriends - enemiesEstimate


                            val reputationCost = (
                                    0.5f +
                                            targetFriendsCount * (0.3f + targetPopularity.coerceAtLeast(0f)) +
                                            neutralsEstimate * 0.3f +
                                            enemiesEstimate * targetPopularity.coerceAtMost(0f) * 0.5f
                                    ).coerceAtLeast(0f)


                            network.valence += enmity + threat - reputationCost
                        }

                        SHARE -> {
                            val recipientNeed = (1f - target.ENERGY * 0.01f).coerceIn(0f, 1f)
                            val surplus = (ratio - 0.6f).coerceAtLeast(0f)
                            val energyCost = 0.1f // 10 energy equivalent
                            val bondGain = 0.2f + max(0f, targetSocialValence) * 0.4f
                            val witnessGain = self.agentsInProximityCount * 0.05f
                            val waste = max(0f, -targetSocialValence) * 0.3f
                            network.valence += surplus * recipientNeed * (bondGain + witnessGain) - energyCost - waste
                        }

                        STEAL -> {
                            val deficit = (0.5f - ratio).coerceAtLeast(0f)
                            val need = deficit * deficit * 4f
                            val greed = 0.1f

                            val targetFriendsCount = target.friendsInProximityCount
                            val targetWitnessCount = target.agentsInProximityCount
                            val targetPopularity = target.localPopularity

                            val nonFriends = targetWitnessCount - targetFriendsCount
                            val enemiesEstimate = nonFriends * ((1f - targetPopularity) / 2f).coerceIn(0f, 1f)
                            val neutralsEstimate = nonFriends - enemiesEstimate


                            val reputationCost = (
                                    baseStealCost +
                                            targetFriendsCount * (witnessStealCost + targetPopularity.coerceAtLeast(0f)) +
                                            neutralsEstimate * witnessStealCost +
                                            enemiesEstimate * targetPopularity.coerceAtMost(0f) * 0.5f
                                    ).coerceAtLeast(-0.15f)

                            network.valence += need + greed - reputationCost
                        }
                    }
                }
            }
        } else {
            when (choice) {
                EAT -> {
                    val food = self.foodField
                    if (food == null) {
                        network.valence -= 0.1f
                        return
                    }

                    val energyRatio = self.ENERGY / MAX_ENERGY
                    self.ENERGY += 10f
                    food.value--

                    val valence = if (energyRatio <= 1f) {
                        (1f - energyRatio).coerceAtLeast(0.5f)
                    } else {
                        -(energyRatio - 1f) * 0.3f
                    }
                    network.valence += valence
                }

                else -> {

                    if (self.TARGET == null) {
                        network.valence -= 0.1f
                        return
                    }
                    val target = self.TARGET as Agent
                    if (!target.enabled) return

                    if (!self.withinActionRadius) {
                        network.valence -= 0.1f
                        self.targetCooldown = 10
                        return
                    }

                    val socialValenceSnapshot = self.globalPopularity

                    when (choice) {
                        MATE -> mate(self, target)
                        KILL -> kill(self, target)
                        SHARE -> share(self, target)
                        STEAL -> steal(self, target)

                    }

                    val valenceDelta = self.globalPopularity - socialValenceSnapshot
                    network.valence += valenceDelta

                }
            }
        }
    }

    fun move(self: Agent, rotate: Float, drive: Float) {
        val movement = drive.coerceIn(-1f, 1f)
        val rotation = rotate.coerceIn(-1f, 1f)
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
            val bystanderToTargetValence = Target.getTargetValence(bystander, target.id)
            val valenceValue =
                if (bystanderToTargetValence > 0) bystanderToTargetValence + 0.3f // killed a friend
                else if (bystanderToTargetValence < 0) bystanderToTargetValence * 0.5f // killed an enemy
                else 0.3f // killed neutral
            witnessValence -= valenceValue
            Target.upsertTargetValence(bystander, self.id, -valenceValue)
            Target.removeTargetFromMemory(bystander, target.id)
        }
        self.globalPopularity -= 0.5f + witnessValence
        Target.removeTargetFromMemory(self, target.id)
        self.targetCooldown = 50

        Target.clearTarget(self)
        if (target.hue == self.hue) Main.killedOwnHue.incrementAndGet()
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
        val myHunger = 1 - (self.ENERGY * INV_MAX_ENERGY)

        if (self.isMale != target.isMale) {

            val targetApproval = target.networkAccess().experimentalTargetAssessment(
                sameHue, oppositeSex, myValenceAtTarget, isIncubating,
                myStanding, myHunger
            )

            if (targetApproval < 0.1) { // Shunned
                Target.upsertTargetValence(self, target.id, -0.1f)
                Target.upsertTargetValence(target, self.id, -0.1f)
                self.targetCooldown = 10
                self.globalPopularity -= 0.1f
                return
            }


            if (self.isMale) { // is male
                if (!target.INCUBATING) {

                    target.INCUBATING = true
                    self.ENERGY -= 20
                    if (self.hue == target.hue) Main.spawnedWithOwnHue.incrementAndGet()
                    else Main.spawnedWithOtherHue.incrementAndGet()
                    target.INCUBATION_TIMER = self.INCUBATION_LENGTH
                    target.INCUBATION_MATERIAL =
                        GeneticMaterial(self.networkAccess().extractWeightsForReproduction(), self.hue!!)

                    Target.upsertTargetValence(self, target.id, 1f)
                    Target.upsertTargetValence(target, self.id, 1f)
                    target.globalPopularity += 1f
                    self.globalPopularity += 1f
                } else self.networkAccess().valence -= 0.2f

            } else if (!self.INCUBATING) {
                self.INCUBATING = true
                target.ENERGY -= 20
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
            } else self.networkAccess().valence -= 0.2f

            Target.upsertTargetValence(self, target.id, 0.5f)
            Target.upsertTargetValence(target, self.id, 0.5f)
            target.globalPopularity += 0.5f
            self.globalPopularity += 0.5f

            self.targetCooldown = 50
            self.networkAccess().valence += 1f
            return

        } else self.networkAccess().valence -= 0.2f

    }

    private fun share(self: Agent, target: Agent) {
        val recipientNeed = (1f - target.ENERGY * 0.01f).coerceIn(0f, 1f)
        target.ENERGY += 10
        self.ENERGY -= 10
        self.networkAccess().valence -= 0.1f // Energy loss equivalent

        // Witnesses
        var witnessValence = 0f
        for (bystander in self.agentsInProximity) {
            if (bystander == null) continue
            if (!bystander.enabled) continue
            Target.upsertTargetValence(bystander, self.id, 0.05f) // helping the community
            witnessValence += 0.05f
        }

        Target.upsertTargetValence(target, self.id, 0.2f * recipientNeed)
        self.globalPopularity += 0.2f * recipientNeed + witnessValence

        if (self.ENERGY <= 0) {
            self.starvation()
        } else Target.upsertTargetValence(self, target.id, 0.05f)
        self.targetCooldown = 10

        if (target.hue == self.hue) Main.sharedWithSameHue.incrementAndGet()
        else Main.sharedWithOtherHue.incrementAndGet()
    }

    private fun steal(self: Agent, target: Agent) {
        target.ENERGY -= 10
        self.ENERGY += 10
        self.networkAccess().valence += 0.1f // Energy gain equivalent

        if (target.ENERGY <= 0) {
            kill(self, target)
            if (target.hue == self.hue) Main.killedOwnHueByStealing.incrementAndGet()
            else Main.killedOtherHueByStealing.incrementAndGet()
            return
        }

        // Witnesses
        var witnessValence = 0f
        for (bystander in self.agentsInProximity) {
            if (bystander == null) continue
            if (!bystander.enabled) continue
            Target.upsertTargetValence(bystander, self.id, -witnessStealCost)
            witnessValence -= 0.1f
        }

        Target.upsertTargetValence(target, self.id, -baseStealCost)

        self.globalPopularity += -baseStealCost + witnessValence
        self.targetCooldown = 10

        if (target.hue == self.hue) Main.stoleFromSameHue.incrementAndGet()
        else Main.stoleFromOtherHue.incrementAndGet()
    }

}
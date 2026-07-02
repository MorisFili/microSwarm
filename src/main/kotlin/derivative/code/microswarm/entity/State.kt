package derivative.code.microswarm.entity

import derivative.code.microswarm.DETECTION_RADIUS
import derivative.code.microswarm.EATING_PROXIMITY
import derivative.code.microswarm.GROUP_SIZE
import derivative.code.microswarm.INV_COUNT
import derivative.code.microswarm.INV_DETECTION_RADIUS
import derivative.code.microswarm.INV_MAX_ENERGY
import derivative.code.microswarm.INV_PRESENCE_CAP
import derivative.code.microswarm.INV_TARGET_RADIUS
import derivative.code.microswarm.MAX_ENERGY
import derivative.code.microswarm.Simulation.Companion.rng
import derivative.code.microswarm.TABLE_SIZE
import derivative.code.microswarm.acosTable
import derivative.code.microswarm.cosTable
import derivative.code.microswarm.network.OutputIndex
import derivative.code.microswarm.sinTable
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sign

object State {

    fun generateEnvironmentState(self: Agent, state: FloatArray) {
        self.friendsInProximityCount = 0

        // ========= Sense of Urgency Block =========

        var acuteScore = 0f
        for (i in 0 until self.targetedBy.size) {
            val other = self.targetedBy[i] ?: continue
            val distance = Target.cheapDistance(self, other) * INV_TARGET_RADIUS
            val distanceScaled = distance * distance
            val otherValence = Target.getTargetValence(self, other.id)
            acuteScore += if (otherValence < 0) {
                abs(otherValence) * (1 - distanceScaled)
            } else 0f
        }

        // ========= Target & Environment Sensing Block =========

        // Local agent analysis logic
        var friendlyX = 0f
        var friendlyY = 0f
        var friendlyCount = 0

        var unFamiliarX = 0f
        var unFamiliarY = 0f
        var unFamiliarCount = 0

        var unFriendlyX = 0f
        var unFriendlyY = 0f
        var unfriendlyCount = 0
        var unfriendlyValue = 0f

        self.localPopularity = 0f

        var crowdEnergy = 0f

        for (agent in self.agentsInProximity) {
            if (agent == null) continue
            if (!agent.enabled) continue
            val dx = agent.x - self.x
            val dy = agent.y - self.y
            val distance = Target.cheapDistance(self, agent) * INV_DETECTION_RADIUS
            val normDistFactor = 1 - distance // 0 = far away, 1 = next to
            val valenceValue = Target.getTargetValence(self, agent.id)
            crowdEnergy += agent.energy

            if (valenceValue > 0f) {            // Positive interactions
                friendlyCount++
                friendlyX += dx
                friendlyY += dy
                self.friendsInProximityCount++
            } else if (valenceValue == 0f) {    // Neutral
                unFamiliarCount++
                unFamiliarX += dx
                unFamiliarY += dy
            } else {                            // Negative interactions
                unFriendlyX += dx
                unFriendlyY += dy
                unfriendlyCount++
                unfriendlyValue += abs(valenceValue) * normDistFactor // Urgency value
            }

            self.localPopularity += Target.getTargetValence(agent, self.id).coerceIn(-1f, 1f)

        }

        // Local resource analysis logic
        var foodValueInProximity = 0f
        var closestFood: Food? = null
        var closestDist = 1000f
        for (entity in self.entitiesInProximity) { // Non-agent entity array
            if (entity is Food) {
                if (!entity.enabled) continue
                val distance = Target.cheapDistance(self, entity) // normalized 0..1
                val ripeness = (entity.value * entity.INV_MAX_VALUE).coerceIn(0f, 1f)
                val availability = (1f + entity.decaySpeed).coerceIn(0f, 1f)
                foodValueInProximity += ripeness * availability

                // Food memory block
                // Only check if the amount of food nodes change between ticks
                // Big performance saver
                if (self.foodInProximityCount > self.recognizedFoodCount) {
                    if (!self.networkAccess().foodEntityId.contains(entity.id)) {
                        val dx = (entity.x - self.x) * INV_DETECTION_RADIUS
                        val dy = (entity.y - self.y) * INV_DETECTION_RADIUS

                        var rotX = dx * -self.facingY + dy * self.facingX
                        var rotY = dx * self.facingX + dy * self.facingY
                        val normalizationVal = Target.cheapDistance(rotX, rotY)
                        if (normalizationVal > 0.0001) {
                            rotX /= normalizationVal
                            rotY /= normalizationVal
                        }
                        val memoryIdx = self.networkAccess().foodRingBuffer
                        self.networkAccess().foodRingBuffer = (memoryIdx + 1) % 3

                        self.networkAccess().foodMemoryRot[memoryIdx] =
                            acosTable[((rotY + 1f) * 0.5f * (TABLE_SIZE - 1)).toInt().coerceIn(0, TABLE_SIZE - 1)] *
                                    if (abs(rotX) > 0f) sign(rotX) else 1f
                        self.networkAccess().foodEntityId[memoryIdx] = entity.id
                    }
                }

                if (distance < closestDist) {
                    closestDist = distance
                    closestFood = entity
                }
            }
        }
        foodValueInProximity *= INV_COUNT[self.foodInProximityCount] // Avg food value
        self.foodValueInProximity = foodValueInProximity
        self.recognizedFoodCount = self.foodInProximityCount

        // Crude gate that checks both for proximity and availability
        // 10px = grazing radius
        self.foodField = if (closestFood != null && closestDist <= EATING_PROXIMITY) closestFood else null
        self.foodFieldDistance = if (closestFood != null) closestDist else 1f

        // ========= Coordination Block =========
        // All normalized to 0 -> +/-1 range
        // Friendly Group
        val friendlyAvgDX =
            if (friendlyCount > 0) friendlyX * INV_COUNT[friendlyCount] * INV_DETECTION_RADIUS else 0f
        val friendlyAvgDY =
            if (friendlyCount > 0) friendlyY * INV_COUNT[friendlyCount] * INV_DETECTION_RADIUS else 0f
        self.metaData[0] = friendlyAvgDX * -self.facingY + friendlyAvgDY * self.facingX
        self.metaData[1] = friendlyAvgDX * self.facingX + friendlyAvgDY * self.facingY
        self.friendlyGroupSize = friendlyCount / GROUP_SIZE
        val friendlyDistance = if (friendlyCount > 0) Target.cheapDistance(friendlyAvgDX, friendlyAvgDY) else 0f
        self.metaData[2] = friendlyDistance

        // Unfamiliar Group
        val unFamAvgDX =
            if (unFamiliarCount > 0) unFamiliarX * INV_COUNT[unFamiliarCount] * INV_DETECTION_RADIUS else 0f
        val unFamAvgDY =
            if (unFamiliarCount > 0) unFamiliarY * INV_COUNT[unFamiliarCount] * INV_DETECTION_RADIUS else 0f
        self.metaData[3] = unFamAvgDX * -self.facingY + unFamAvgDY * self.facingX
        self.metaData[4] = unFamAvgDX * self.facingX + unFamAvgDY * self.facingY
        val unFamiliarGroupSize = unFamiliarCount / GROUP_SIZE
        val unFamiliarDistance = if (unFamiliarCount > 0) {
            Target.cheapDistance(unFamAvgDX, unFamAvgDY) * unFamiliarGroupSize
        } else 0f
        self.metaData[5] = unFamiliarDistance

        // Unfriendly Group
        val unfriendlyAvgDX =
            if (unfriendlyCount > 0) unFriendlyX * INV_COUNT[unfriendlyCount] * INV_DETECTION_RADIUS else 0f
        val unfriendlyAvgDY =
            if (unfriendlyCount > 0) unFriendlyY * INV_COUNT[unfriendlyCount] * INV_DETECTION_RADIUS else 0f
        self.metaData[6] = unfriendlyAvgDX * -self.facingY + unfriendlyAvgDY * self.facingX
        self.metaData[7] = unfriendlyAvgDX * self.facingX + unfriendlyAvgDY * self.facingY
        val unfriendlyCenterStrength =
            if (unfriendlyCount > 0) unfriendlyValue * INV_COUNT[unfriendlyCount] else 0f
        val unfriendlyDistance = if (unfriendlyCount > 0) {
            Target.cheapDistance(unfriendlyAvgDX, unfriendlyAvgDY) * unfriendlyCenterStrength
        } else 0f
        self.metaData[8] = unfriendlyDistance

        // Resources
        val resourceDX = if (closestFood != null) (closestFood.x - self.x) * INV_DETECTION_RADIUS else 0f
        val resourceDY = if (closestFood != null) (closestFood.y - self.y) * INV_DETECTION_RADIUS else 0f
        self.metaData[12] = resourceDX * -self.facingY + resourceDY * self.facingX
        self.metaData[13] = resourceDX * self.facingX + resourceDY * self.facingY
        val resourceDistance = if (closestFood != null)
            Target.cheapDistance(self, closestFood) * INV_DETECTION_RADIUS else 0f
        self.metaData[14] = resourceDistance

        // Environmental
        val foodAvailability = if (closestFood != null) foodValueInProximity else 0f
        val hunger = (1 - (self.energy / MAX_ENERGY)).coerceIn(0f, 1f)
        val immediateThreat = acuteScore.coerceIn(0f, 1f)
        val crowdingState = self.agentsInProximityCount * INV_PRESENCE_CAP
        self.threatState = ((unfriendlyCenterStrength * unfriendlyDistance) + immediateThreat).coerceIn(0f, 1f)
        val localPopularity = self.localPopularity * INV_COUNT[self.agentsInProximityCount]
        self.localAgentEnergyValue = if (self.agentsInProximityCount > 0)
            crowdEnergy * INV_COUNT[self.agentsInProximityCount] * 0.01f else 0f







        // ========= State Assignment Block =========
        // same hue group coordination
        state[Index.FRIEND_DIST] = friendlyDistance.coerceIn(0f, 1f)
        state[Index.FRIEND_GROUP_SIZE] = self.friendlyGroupSize.coerceIn(0f, 1f)
        // other hue group coordination
        state[Index.UNFAM_DIST] = unFamiliarDistance.coerceIn(0f, 1f)
        state[Index.UNFAM_SIZE] = unFamiliarGroupSize.coerceIn(0f, 1f)
        // renegade group coordination
        state[Index.HOSTILE_DIST] = unfriendlyDistance.coerceIn(0f, 1f)
        state[Index.HOSTILE_STR] = unfriendlyCenterStrength.coerceIn(0f, 1f)
        // resource coordination
        state[Index.RESOURCE_DIST] = resourceDistance.coerceIn(0f, 1f)
        state[Index.RESOURCE_DENSITY] = foodAvailability.coerceIn(0f, 1f)
        // global environmental && self
        state[Index.IN_FOOD_FIELD] = if (self.foodField != null) 1f else 0f
        state[Index.CROWDING] = crowdingState.coerceIn(-1f, 1f)
        state[Index.THREAT] = self.threatState.coerceIn(-1f, 1f)
        state[Index.IS_INCUBATING] = if (self.INCUBATING) 1f else 0f
        state[Index.HUNGER] = hunger.coerceIn(0f, 1f)
        state[Index.POPULARITY] = localPopularity.coerceIn(-1f, 1f)

    }

    fun updateTransmitters(self: Agent) {
        val foodAvailability = self.foodValueInProximity
        val hunger = (1 - (self.energy / MAX_ENERGY)).coerceIn(0f, 1f)
        val mattersLessWhenHungry = (1 - hunger) * (1 - hunger)
        val hungryWithNoFoodAround = hunger * (1f - foodAvailability.coerceIn(0f, 1f))
        // val hungryWithFoodAround = hunger * foodAvailability.coerceIn(0f, 1f)
        val localPopularity = self.localPopularity * INV_COUNT[self.agentsInProximityCount]

        var d0 = self.stateEMA[self.ownEnergyDelta]
        d0 += ((self.energy - self.PRE_ENERGY_SNAP) * INV_MAX_ENERGY - d0) * 0.2f
        if (abs(d0) < 1e-6f) d0 = 0f
        self.stateEMA[self.ownEnergyDelta] = d0

        var d1 = self.stateEMA[self.hostileDistDelta]
        d1 += ((self.hostileDistance - self.PRE_HOSTILE_DIST) - d1) * 0.2f
        if (abs(d1) < 1e-6f) d1 = 0f
        self.stateEMA[self.hostileDistDelta] = d1

        var d2 = self.stateEMA[self.groupStress]
        d2 += ((1f - self.localAgentEnergyValue) - d2) * 0.1f
        if (abs(d2) < 1e-6f) d2 = 0f
        self.stateEMA[self.groupStress] = d2

        var d3 = self.stateEMA[self.familiarRatio]
        d3 += (self.friendlyGroupSize * INV_COUNT[self.agentsInProximityCount.coerceAtLeast(1)] - d3) * 0.1f
        if (abs(d3) < 1e-6f) d3 = 0f
        self.stateEMA[self.familiarRatio] = d3

        self.PRE_HOSTILE_DIST = self.hostileDistance
        self.PRE_ENERGY_SNAP = self.energy

        val ownEnergyDelta = self.stateEMA[self.ownEnergyDelta]
        val hostileDistDelta = self.stateEMA[self.hostileDistDelta]
        val groupStress = self.stateEMA[self.groupStress]
        val familiarRatio = self.stateEMA[self.familiarRatio]

        // ADRENALINE - Urgency/Intensity signal
        val energyFalling = (-ownEnergyDelta).coerceIn(0f, 1f)        // only the negative side
        val threatClosing = (-hostileDistDelta).coerceIn(0f, 1f)       // only when getting closer
        val groupIsStressed = groupStress.coerceIn(0f, 1f)
        val socialHostility = max(0f, -localPopularity)
        val crowdIsUnfamiliar = (1f - familiarRatio).coerceIn(0f, 1f)
        val targetedByTargetThreatUrgency = self.metaData[16] * (1f - self.targetDistance).coerceIn(0f, 1f)
        val dangerAxis = self.threatState.coerceIn(0f, 1f)

        val adrenaline = (
                energyFalling +
                        threatClosing +
                        groupIsStressed +
                        (socialHostility * mattersLessWhenHungry) +
                        crowdIsUnfamiliar * 0.5f +
                        hungryWithNoFoodAround +
                        targetedByTargetThreatUrgency +
                        dangerAxis
                )
        self.neurotransmitters[0] = adrenaline


        // SEROTONIN - Satisfaction/Content signal
        val energyRising = ownEnergyDelta.coerceIn(0f, 1f)
        val threatReceding = hostileDistDelta.coerceIn(0f, 1f)
        val groupIsCalm = (1f - groupStress).coerceIn(0f, 1f)
        val socialWarmth = (max(0f, localPopularity) * mattersLessWhenHungry)
        val crowdIsFamiliar = familiarRatio.coerceIn(0f, 1f)
        val feelingSated = (1f - hunger).coerceIn(0f, 1f)

        val serotonin = (
                energyRising +
                        threatReceding +
                        groupIsCalm +
                        socialWarmth +
                        crowdIsFamiliar * 0.5f +
                        feelingSated
                )
        self.neurotransmitters[1] = serotonin
    }

    fun intentToSpatialTransformation(self: Agent, state: FloatArray) {

        val friendlyCenterX = self.metaData[0]
        val friendlyCenterY = self.metaData[1]
        val friendlyDistance = self.metaData[2]

        val unFamiliarCenterX = self.metaData[3]
        val unFamiliarCenterY = self.metaData[4]
        val unFamiliarDistance = self.metaData[5]

        val unfriendlyCenterX = self.metaData[6]
        val unfriendlyCenterY = self.metaData[7]
        val unfriendlyCenterThreat = self.metaData[8]

        val targetX = self.metaData[9]
        val targetY = self.metaData[10]
        val targetDistance = self.metaData[11]

        val resourceX = self.metaData[12]
        val resourceY = self.metaData[13]
        val resourceDistance = self.metaData[14]

        // Softmax-Argmax logic for Intent
        //
        // 7 ->   TARGET
        // 8 ->   FOOD
        // 9 -> FRIENDLY
        // 10 -> UNFAMILIAR
        // 11 -> UNFRIENDLY

        var intentDistance = Float.NaN
        var alignmentError = Float.NaN

        var intentTarget = self.output[OutputIndex.TARGET]
        var intentFood = self.output[OutputIndex.FOOD]
        var intentFam = self.output[OutputIndex.FRIEND]
        var intentUnFam = self.output[OutputIndex.UNFAMILIAR]
        val intentDanger = self.output[OutputIndex.UNFRIENDLY]

        var invalidTarget = false
        val selectedIntent = self.networkAccess().committedSpatialIntent

        when (selectedIntent) {
            1 -> {
                invalidTarget = targetDistance == 0f
                intentFood = 0f
                intentFam = 0f
                intentUnFam = 0f
            }

            2 -> {
                invalidTarget = resourceDistance == 0f
                intentTarget = 0f
                intentFam = 0f
                intentUnFam = 0f
            }

            3 -> {
                invalidTarget = friendlyDistance == 0f
                intentTarget = 0f
                intentFood = 0f
                intentUnFam = 0f
            }

            4 -> {
                invalidTarget = unFamiliarDistance == 0f
                intentTarget = 0f
                intentFood = 0f
                intentFam = 0f
            }
        }

        if (selectedIntent != 0 && !invalidTarget) {

            // Intent translation
            var steerX = (intentTarget * targetX) + (intentFood * resourceX) +
                    (intentFam * friendlyCenterX) +
                    (intentUnFam * unFamiliarCenterX) + (intentDanger * unfriendlyCenterX)

            var steerY = (intentTarget * targetY) + (intentFood * resourceY) +
                    (intentFam * friendlyCenterY) +
                    (intentUnFam * unFamiliarCenterY) + (intentDanger * unfriendlyCenterY)

            val normalizedTargetDistance = targetDistance * 2f

            val wTarget = abs(intentTarget)
            val wFood = abs(intentFood)
            val wFam = abs(intentFam)
            val wUnFam = abs(intentUnFam)
            val wDanger = abs(intentDanger)
            val wSum = wTarget + wFood + wFam + wUnFam + wDanger

            intentDistance = if (wSum > 1e-4f) (
                    wTarget * (if (intentTarget >= 0f) normalizedTargetDistance else 1f - normalizedTargetDistance) +
                            wFood * (if (intentFood >= 0f) resourceDistance else 1f - resourceDistance) +
                            wFam * (if (intentFam >= 0f) friendlyDistance else 1f - friendlyDistance) +
                            wUnFam * (if (intentUnFam >= 0f) 1f - unFamiliarDistance else unFamiliarDistance) +
                            wDanger * (if (intentDanger >= 0f) 1f - unfriendlyCenterThreat else unfriendlyCenterThreat)
                    ) / wSum else 0f

            val steerMagnitude = Target.cheapDistance(steerX, steerY)
            if (steerMagnitude > 0.0001f) {
                steerX /= steerMagnitude
                steerY /= steerMagnitude
            }

            alignmentError =
                acosTable[((steerY + 1f) * 0.5f * (TABLE_SIZE - 1)).toInt().coerceIn(0, TABLE_SIZE - 1)] *
                        if (abs(steerX) > 0f) sign(steerX) else 1f

            // Clear old exploration target
            self.explorationTargetX = 0f
            self.explorationTargetY = 0f

        } else {
            // Exploration block

            if (invalidTarget) {
                // Check memory
                if (selectedIntent == 2) { // Food
                    val memoryIdx = (self.networkAccess().foodRingBuffer + 2) % 3
                    val memoryRot = self.networkAccess().foodMemoryRot[memoryIdx]
                    if (!memoryRot.isNaN()) {
                        alignmentError = memoryRot

                        val angleIndex = (((alignmentError + 1) / 2) * TABLE_SIZE).toInt().coerceIn(0, TABLE_SIZE - 1)
                        val cosA = cosTable[angleIndex]
                        val sinA = sinTable[angleIndex]

                        self.explorationTargetX = self.x + (self.facingX * cosA - self.facingY * sinA) * 35f
                        self.explorationTargetY = self.y + (self.facingX * sinA + self.facingY * cosA) * 35f
                    }
                }
            }

            self.networkAccess().committedSpatialIntent = 0

            // Out of range check
            val distance = Target.cheapDistance(self.explorationTargetX - self.x, self.explorationTargetY - self.y)
            if (distance > DETECTION_RADIUS) {
                self.explorationTargetX = 0f
                self.explorationTargetY = 0f
            }

            if (self.explorationTargetX == 0f && self.explorationTargetY == 0f) {
                val angle = rng.nextFloat(0.4f, 0.6f) * TABLE_SIZE
                val cos = cosTable[angle.toInt()]
                val sin = sinTable[angle.toInt()]
                val right = sin + intentDanger * unfriendlyCenterX // Urgency heading injection
                val forward = cos + intentDanger * unfriendlyCenterY // Urgency heading injection
                self.explorationTargetX = self.x + (self.facingX * forward - self.facingY * right) * 35f
                self.explorationTargetY = self.y + (self.facingX * right + self.facingY * forward) * 35f
            }

            val explDX = (self.explorationTargetX - self.x) * INV_DETECTION_RADIUS
            val explDY = (self.explorationTargetY - self.y) * INV_DETECTION_RADIUS

            val explorationDist = Target.cheapDistance(explDX, explDY)

            var explX = (explDX * -self.facingY + explDY * self.facingX)
            var explY = (explDX * self.facingX + explDY * self.facingY)
            val explorationMag = Target.cheapDistance(explX, explY)

            if (explorationMag > 0.0001f) {
                explX /= explorationMag
                explY /= explorationMag
                alignmentError =
                    acosTable[((explY + 1f) * 0.5f * (TABLE_SIZE - 1)).toInt().coerceIn(0, TABLE_SIZE - 1)] *
                            if (abs(explX) > 0f) sign(explX) else 1f
            } else alignmentError = 0f

            val dangerUrgency = max(0f, -intentDanger) * unfriendlyCenterThreat
            intentDistance = explorationDist - dangerUrgency

            if (explorationDist < 0.1f) {
                // Target reached
                self.explorationTargetX = 0f
                self.explorationTargetY = 0f
            }
        }

        state[0] = alignmentError.coerceIn(-1f, 1f)
        state[1] = intentDistance.coerceIn(0f, 1f)

    }


    object Index {
        // State Mapping
        const val FRIEND_DIST = 0
        const val FRIEND_GROUP_SIZE = 1
        const val UNFAM_DIST = 2
        const val UNFAM_SIZE = 3
        const val HOSTILE_DIST = 4
        const val HOSTILE_STR = 5
        const val RESOURCE_DIST = 6
        const val RESOURCE_DENSITY = 7
        const val TARGET_DIST = 8
        const val TARGET_HUE = 9
        const val TARGET_SEX = 10
        const val TARGET_VALENCE = 11
        const val TARGET_IS_INCUB = 12
        const val TARGET_POPULARITY = 13
        const val TARGET_HUNGER = 14
        const val TARGET_THREAT = 15
        const val WITHIN_ACTION_RADIUS = 16
        const val IN_FOOD_FIELD = 17
        const val CROWDING = 18
        const val THREAT = 19
        const val IS_INCUBATING = 20
        const val HUNGER = 21
        const val POPULARITY = 22

        // Feedback
        const val A_KILL = 23
        const val A_MATE = 24
        const val A_EAT = 25
        const val A_SHARE = 26
        const val A_STEAL = 27
        const val S_TARGET = 28
        const val S_FOOD = 29
        const val S_FRIEND = 30
        const val S_UNFAMILIAR = 31
        const val P_SEX = 32
        const val P_VALENCE = 33
        const val P_HEALTH = 34


        val COUNT = Index::class.java.declaredFields.count { it.type == Int::class.java }
    }
}


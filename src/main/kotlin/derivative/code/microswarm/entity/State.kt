package derivative.code.microswarm.entity

import derivative.code.microswarm.DETECTION_RADIUS
import derivative.code.microswarm.GROUP_SIZE
import derivative.code.microswarm.INV_COUNT
import derivative.code.microswarm.INV_DETECTION_RADIUS
import derivative.code.microswarm.INV_MAX_ENERGY
import derivative.code.microswarm.INV_TARGET_RADIUS
import derivative.code.microswarm.MAX_ENERGY
import derivative.code.microswarm.Simulation.Companion.rng
import derivative.code.microswarm.TABLE_SIZE
import derivative.code.microswarm.TARGET_RADIUS
import derivative.code.microswarm.acosTable
import derivative.code.microswarm.cosTable
import derivative.code.microswarm.sinTable
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sign

object State {

    fun generateState(self: Agent, state: FloatArray) {
        self.friendsInProximityCount = 0

        // ========= Sense of Urgency Block =========

        var acuteScore = 0f
        for (i in 0 until self.targetedBy.size) {
            val other = self.targetedBy[i] ?: continue
            val distance = Target.cheapDistance(self, other) * INV_TARGET_RADIUS
            val distanceScaled = distance * distance
            val otherValence = Target.getTargetValence(self, other.id)
            acuteScore += if (otherValence < 0) {
                abs(otherValence) * (1 - distanceScaled) * 5f // -0.2 Valence standing right next to you = urgent
            } else 0f
        }

        // ========= Target & Environment Sensing Block =========

        // Target clearing logic
        if (self.targetCooldown > 0) Target.clearTarget(self)
        val target = self.TARGET // Local pointer
        if (target != null && !target.enabled) Target.clearTarget(self) // Clear valid but disabled target
        if (target != null) { // Old target that went out of range
            val targetDist = Target.cheapDistance(self, target)
            if (targetDist > TARGET_RADIUS) Target.clearTarget(self)
        }

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
            crowdEnergy += agent.ENERGY

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
                unfriendlyValue += abs(valenceValue) * normDistFactor * 5f // Urgency value
            }

            self.localPopularity += Target.getTargetValence(agent, self.id)

        }

        self.withinActionRadius = Target.actionRadiusCheck(self, self.TARGET)

        // Local resource analysis logic
        var foodValue = 0f
        var closestFood: Food? = null
        var closestDist = 1000f
        for (entity in self.entitiesInProximity) { // Non-agent entity array
            if (entity is Food) {
                if (!entity.enabled) continue
                val distance = Target.cheapDistance(self, entity) * INV_DETECTION_RADIUS // normalized 0..1
                val prox = (1f - distance).coerceIn(0f, 1f)
                val ripeness = (entity.value / 25f).coerceIn(0f, 1f)
                val availability = (1f + entity.decaySpeed).coerceIn(0f, 1f)
                foodValue += prox * ripeness * availability

                if (distance < closestDist) {
                    closestDist = distance
                    closestFood = entity
                }
            }
        }
        // Crude gate that checks both for proximity and availability
        // 10px = grazing radius
        self.foodField = if (closestFood != null && closestDist < 0.25f) closestFood else null
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
        val friendlyGroupSize = friendlyCount / GROUP_SIZE
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
        val unFamiliarDiscomfort = if (unFamiliarCount > 0) {
            (1 - Target.cheapDistance(unFamAvgDX, unFamAvgDY)) * unFamiliarGroupSize
        } else 0f
        self.metaData[5] = unFamiliarDiscomfort

        // Unfriendly Group
        val unfriendlyAvgDX =
            if (unfriendlyCount > 0) unFriendlyX * INV_COUNT[unfriendlyCount] * INV_DETECTION_RADIUS else 0f
        val unfriendlyAvgDY =
            if (unfriendlyCount > 0) unFriendlyY * INV_COUNT[unfriendlyCount] * INV_DETECTION_RADIUS else 0f
        self.metaData[6] = unfriendlyAvgDX * -self.facingY + unfriendlyAvgDY * self.facingX
        self.metaData[7] = unfriendlyAvgDX * self.facingX + unfriendlyAvgDY * self.facingY
        val unfriendlyCenterStrength =
            if (unfriendlyCount > 0) unfriendlyValue * INV_COUNT[unfriendlyCount] else 0f
        val unfriendlyCenterThreat = if (unfriendlyCount > 0) {
            (1f - Target.cheapDistance(unfriendlyAvgDX, unfriendlyAvgDY)) * unfriendlyCenterStrength
        } else 0f
        self.metaData[8] = unfriendlyCenterThreat

        // Target
        val targetDX = if (target != null) (target.x - self.x) * INV_TARGET_RADIUS else 0f
        val targetDY = if (target != null) (target.y - self.y) * INV_TARGET_RADIUS else 0f
        self.metaData[9] = targetDX * -self.facingY + targetDY * self.facingX
        self.metaData[10] = targetDX * self.facingX + targetDY * self.facingY
        self.targetDistance = if (target != null) Target.cheapDistance(self, target) * INV_TARGET_RADIUS else 0f
        self.metaData[11] = self.targetDistance
        val targetValence = if (target != null) Target.getTargetValence(self, target.id) else 0f
        val targetPopularity = if (target != null) target.localPopularity else 0f
        val targetHunger = if (target != null) 1 - (target.ENERGY * INV_MAX_ENERGY) else 0f

        self.targetAttractiveness = if (target != null) {
            val sexCompat = if (target.isMale != self.isMale) 1f else 0f
            val health = (target.ENERGY * INV_MAX_ENERGY).coerceIn(0f, 1f)
            val available = if (!target.INCUBATING) 1f else 0f
            sexCompat * 0.5f + health * 0.15f + available * 0.1f
        } else 0f

        self.threatFromTarget = if (target != null) {
            val valence = if (targetValence < 0) abs(targetValence) else 0f
            val huntingMe = if (self.targetedBy.any { it === target }) 1f else 0f
            (huntingMe * valence).coerceIn(0f, 1f)
        } else 0f


        // Resources
        val resourceDX = if (closestFood != null) (closestFood.x - self.x) * INV_DETECTION_RADIUS else 0f
        val resourceDY = if (closestFood != null) (closestFood.y - self.y) * INV_DETECTION_RADIUS else 0f
        self.metaData[12] = resourceDX * -self.facingY + resourceDY * self.facingX
        self.metaData[13] = resourceDX * self.facingX + resourceDY * self.facingY
        val resourceDistance = if (closestFood != null)
            Target.cheapDistance(self, closestFood) * INV_DETECTION_RADIUS else 0f
        self.metaData[14] = resourceDistance

        // Environmental
        val incubationMultiplier = if (self.INCUBATING) 2f else 1f
        val competitors = (friendlyCount + unFamiliarCount) / GROUP_SIZE
        val resourceDensity = foodValue * INV_COUNT[1 + competitors.toInt()]
        val hunger = (1 - (self.ENERGY / MAX_ENERGY)).coerceIn(0f, 1f)
        val immediateThreat = acuteScore.coerceIn(0f, 1f)
        val crowdingState = (((friendlyCount + unFamiliarCount) - GROUP_SIZE) / GROUP_SIZE)
        val threatState = ((unfriendlyCenterStrength * unfriendlyCenterThreat) + immediateThreat).coerceIn(0f, 1f)
        val discomfortState = unFamiliarGroupSize * unFamiliarDiscomfort
        val explorationPressure = (
                (1f - resourceDensity.coerceIn(0f, 1f)) * 0.75f +
                        (1f - friendlyGroupSize.coerceIn(0f, 1f)) * 0.25f) *
                (1f - (unfriendlyCenterThreat + unFamiliarDiscomfort * (1f - hunger)).coerceIn(0f, 1f))
        val safetyScore = max(
            ((1f - friendlyDistance) * friendlyGroupSize) - (unFamiliarDiscomfort * unFamiliarGroupSize), // social safety
            (1f - threatState) * (1f - explorationPressure)
        )  // environmental safety

        val popularitySquared = abs(self.localPopularity) * self.localPopularity
        val localAgentEnergyValue = if (self.agentsInProximityCount > 0)
            crowdEnergy * INV_COUNT[self.agentsInProximityCount] * 0.01f else 0f


        // Main signal evaluation
        val mattersLessWhenHungry = (1 - hunger) * (1 - hunger)
        // ADRENALINE - Urgency/Intensity signal
        val dangerAxis = threatState.coerceIn(0f, 1f) * (1 - safetyScore.coerceIn(0f, 1f))
        val discomfortAxis = discomfortState.coerceIn(0f, 1f) * (1 - safetyScore.coerceIn(0f, 1f))
        val crowdingWithNoFood = crowdingState.coerceIn(0f, 1f) * (1f - resourceDensity.coerceIn(0f, 1f))
        val threatenedAndVulnerable = immediateThreat * incubationMultiplier
        val hungryWithNoFoodAround = hunger * (1f - resourceDensity.coerceIn(0f, 1f))
        val targetedByThreatUrgency = self.threatFromTarget * (1f - self.targetDistance)
        val dislikedByOthers = max(0f, -popularitySquared).coerceAtMost(0.65f)

        val adrenaline = (
            (discomfortAxis + dislikedByOthers) * mattersLessWhenHungry +
                crowdingWithNoFood + threatenedAndVulnerable + hungryWithNoFoodAround +
                    targetedByThreatUrgency + dangerAxis
                ).coerceIn(0f, 1f)
        self.neurotransmitters[0] = adrenaline


        // DOPAMINE - Appetite/Desire signal
        val attractiveTargetUrgency = self.targetAttractiveness * (1 - self.targetDistance)
        val hungryWithFoodAround = hunger * resourceDensity.coerceIn(0f, 1f)
        val stickingWithFriends = friendlyGroupSize * (1f - friendlyDistance) * 0.5f * incubationMultiplier
        val foragingWhenNoFoodAround = hunger * (1f - resourceDensity.coerceIn(0f, 1f)) *
                (self.agentsInProximityCount * 0.03125f).coerceAtMost(1f) * localAgentEnergyValue

        val dopamine = (
            (attractiveTargetUrgency + stickingWithFriends) * mattersLessWhenHungry +
                hungryWithFoodAround + foragingWhenNoFoodAround + explorationPressure.coerceIn(0f, 1f)
                ).coerceIn(0f, 1f)
        self.neurotransmitters[1] = dopamine

        // SEROTONIN - Satisfaction/Content signal
        val feelingSafe = safetyScore.coerceIn(0f, 1f) * 0.35f
        val feelingSated = (1f - hunger).coerceIn(0f, 1f) * 0.35f
        val resourceAvailability = resourceDensity.coerceIn(0f, 1f) * 0.20f
        val likedByOthers = max(0f, popularitySquared).coerceAtMost(0.65f)

        val serotonin = (
                (feelingSafe + likedByOthers) * mattersLessWhenHungry +
                feelingSated + resourceAvailability
                ).coerceIn(0f, 1f)
        self.neurotransmitters[2] = serotonin

        // ========= State Assignment Block =========
        // same hue group coordination
        state[0] = friendlyDistance.coerceIn(0f, 1f)
        state[1] = friendlyGroupSize.coerceIn(0f, 1f)
        // other hue group coordination
        state[2] = unFamiliarDiscomfort.coerceIn(0f, 1f)
        state[3] = unFamiliarGroupSize.coerceIn(0f, 1f)
        // renegade group coordination
        state[4] = unfriendlyCenterThreat.coerceIn(0f, 1f)
        state[5] = unfriendlyCenterStrength.coerceIn(0f, 1f)
        // resource coordination
        state[6] = resourceDistance.coerceIn(0f, 1f)
        // target coordination
        state[7] = self.targetDistance.coerceIn(0f, 1f)
        // target perception
        state[8] = if (target == null) 0f else if (target.hue == self.hue) 1f else -1f
        state[9] = if (target == null) 0f else if (target.isMale != self.isMale) 1f else -1f
        state[10] = if (target == null) 0f else targetValence.coerceIn(-1f, 1f) // target relationship
        state[11] = if (target == null) 0f else if (target.INCUBATING) -1f else 1f
        state[12] = if (target == null) 0f else targetPopularity.coerceIn(0f, 1f) // local social standing
        state[24] = targetHunger.coerceIn(0f, 1f)
        // global environmental && self
        state[13] = if (self.withinActionRadius) 1f else 0f
        state[14] = if (self.foodField != null) 1f else 0f
        state[15] = crowdingState.coerceIn(-1f, 1f)
        state[16] = threatState.coerceIn(-1f, 1f)
        state[17] = if (self.INCUBATING) 1f else 0f
        state[18] = safetyScore.coerceIn(0f, 1f)
        state[19] = resourceDensity.coerceIn(0f, 1f)
        state[20] = explorationPressure.coerceIn(0f, 1f)
        state[21] = hunger.coerceIn(0f, 1f)
        state[22] = discomfortState.coerceIn(0f, 1f)
        state[23] = popularitySquared.coerceIn(-1f, 1f)

    }

    fun generateIntentState(self: Agent, state: FloatArray) {

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

        var intentDistance = 1f
        var alignmentError = 0f

        var intentTarget = self.output[7]
        var intentFood = self.output[8]
        var intentFam = self.output[9]
        var intentUnFam = self.output[10]
        val intentDanger = self.output[11]

        var invalidTarget = false

        when (self.networkAccess().committedSpatialIntent) {
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

        if (self.networkAccess().committedSpatialIntent != 0 && !invalidTarget) {

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
}

// State Mapping
const val FRIEND_DIST = 0
const val FRIEND_GROUP_SIZE = 1
const val UNFAM_DIST = 2
const val UNFAM_SIZE = 3
const val HOSTILE_DIST = 4
const val HOSTILE_STR = 5
const val RESOURCE_DIST = 6
const val TARGET_DIST = 7
const val TARGET_HUE = 8
const val TARGET_SEX = 9
const val TARGET_VALENCE = 10
const val TARGET_IS_INCUB = 11
const val TARGET_POPULARITY = 12
const val TARGET_HUNGER = 24
const val WITHIN_ACTION_RADIUS = 13
const val IN_FOOD_FIELD = 14
const val CROWDING = 15
const val THREAT = 16
const val IS_INCUBATING = 17
const val SAFETY_SCORE = 18
const val RESOURCE_DENSITY = 19
const val EXP_PRESSURE = 20
const val HUNGER = 21
const val DISCOMFORT = 22
const val POPULARITY_SQ = 23
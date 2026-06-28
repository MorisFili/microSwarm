package derivative.code.microswarm.entity

import derivative.code.microswarm.ACTION_RADIUS
import derivative.code.microswarm.INV_TARGET_RADIUS
import derivative.code.microswarm.TARGET_RADIUS
import derivative.code.microswarm.network.OutputIndex
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sign

object Target {

    fun selectTarget(self: Agent) {

        if (self.targetCooldown > 0) return

        // 20,21 ->   SEX -> (1) OPPOSITE (-1) SAME
        // 22,23 ->   VALENCE -> (1) POSITIVE (-1) NEGATIVE
        // 24,25 ->   HEALTH -> (1) HIGHEST (-1) LOWEST
        val sexPref = sign(self.output[OutputIndex.SEX])
        val sexMag = abs(self.output[OutputIndex.SEX])
        val valencePref = sign(self.output[OutputIndex.VALENCE])
        val valenceMag = abs(self.output[OutputIndex.VALENCE])
        val healthPref = sign(self.output[OutputIndex.HEALTH])
        val healthMag = abs(self.output[OutputIndex.HEALTH])

        // ========= Targeted By Block =========
        var bestTargetedDistance = 1000f
        var closestTargetedBy: Agent? = null
        for (i in 0 until self.targetedBy.size) {
            val other = self.targetedBy[i] ?: continue
            if (!other.enabled) continue
            val distance = cheapDistance(self, other) * INV_TARGET_RADIUS
            val distanceScaled = distance * distance
            val otherValence = getTargetValence(self, other.id)
            val urgency = if (otherValence < 0) abs(otherValence) * (1 - distanceScaled) else 0f
            // 0.09 urgency = 0.1 valence within action radius
            if (urgency > 0.09f && distance < bestTargetedDistance && lineOfSightCheck(self, other)) {
                closestTargetedBy = other
                bestTargetedDistance = distance
            }
        }

        // ======== Possible targets ========
        var targetScore = 0f
        var pickedTarget: Agent? = null
        var closestDist = 1000f
        var closestTarget: Agent? = null
        for (other in self.agentsInProximity) {
            if (other == null) continue
            if (!other.enabled) continue
            if (lineOfSightCheck(self, other)) {

                var score = 0f
                val targetValence = getTargetValence(self, other.id)

                if (self.isMale != other.isMale && sexPref > 0) score += sexMag
                if (self.isMale == other.isMale && sexPref < 0) score += sexMag
                if (valencePref > 0) score += targetValence * valenceMag
                if (valencePref < 0) score -= targetValence * valenceMag
                if (healthPref > 0) score += ((other.energy - 50f) * 0.01f) * healthMag
                if (healthPref < 0) score += ((50f - other.energy) * 0.01f) * healthMag

                if (score > targetScore) {
                    pickedTarget = other
                    targetScore = score
                }

                val targetDist = cheapDistance(self, other)
                if (targetDist < closestDist) {
                    closestDist = targetDist
                    closestTarget = other
                }
            }
        }

        // Self defense preference
        if (closestTargetedBy != null) {
            setTarget(self, closestTargetedBy)
            return
        }

        if (pickedTarget != null) {
            setTarget(self, pickedTarget)
            return
        }

        if (closestTarget != null) {
            setTarget(self, closestTarget)
        }

        if (self.TARGET != null) self.targetPopularity = self.TARGET!!.localPopularity
        else self.targetPopularity = 0f
    }


    fun getTargetValence(self: Agent, targetId: Int): Float {
        val startIndex = targetId and 127
        var step = 0
        while (step < 128) {
            val index = (startIndex + step) and 127
            if (self.interactionId[index] == targetId) return self.interactionValence[index]
            if (self.interactionId[index] == -1) return 0f
            step++
        }
        return 0.0f
    }

    fun upsertTargetValence(self: Agent, targetId: Int, valence: Float) {
        // RW model based updates
        val startIndex = targetId and 127
        var step = 0
        var emptyIndex = -1
        while (step < 128) {
            val index = (startIndex + step) and 127
            if (self.interactionId[index] == targetId) { // update
                val iv = self.interactionValence[index]
                self.interactionValence[index] = iv + abs(valence) * (sign(valence) - iv)
                return
            }
            if (emptyIndex == -1 && self.interactionValence[index] == 0f) emptyIndex = index
            if (self.interactionId[index] == -1) { // reached the end - insert
                val insertIndex = if (emptyIndex != -1) emptyIndex else index
                self.interactionId[insertIndex] = targetId
                val iv = self.interactionValence[insertIndex]
                self.interactionValence[insertIndex] = iv + abs(valence) * (sign(valence) - iv)
                return
            }
            step++
        }
        // replace the weakest fallback
    }

    fun removeTargetFromMemory(self: Agent, targetId: Int) {
        val startIndex = targetId and 127
        var step = 0
        while (step < 128) {
            val index = (startIndex + step) and 127
            if (self.interactionId[index] == targetId) { // ID overwritten in upsert function
                self.interactionValence[index] = 0f
                return
            }
            if (self.interactionId[index] == -1) return
            step++
        }
    }

    fun actionRadiusCheck(self: Agent, target: Agent?): Boolean {
        if (target == null) return false
        if (!target.enabled) return false

        val dx = target.x - self.x
        val dy = target.y - self.y
        val targetDistSq = dx * dx + dy * dy
        val actionRadiusSq = ACTION_RADIUS * ACTION_RADIUS
        if (targetDistSq <= actionRadiusSq && lineOfSightCheck(self, target)) {
            return true
        } else return false
    }

    fun lineOfSightCheck(self: Agent, target: Agent): Boolean {
        if (!target.enabled) return false

        val dx = target.x - self.x
        val dy = target.y - self.y
        val targetDistSq = dx * dx + dy * dy
        val targetRadiusSq = TARGET_RADIUS * TARGET_RADIUS
        if (targetDistSq > 0.000001f && targetDistSq <= targetRadiusSq) {
            val moveLenSq = self.facingX * self.facingX + self.facingY * self.facingY
            if (moveLenSq > 0.000001f) {
                val forward = self.facingX * dx + self.facingY * dy
                return forward > 0 && forward * forward > targetDistSq * moveLenSq * 0.25f
            } else return false
        } else return false
    }

    fun setTarget(self: Agent, target: Agent?) {
        if (self.TARGET != null) clearTarget(self)
        self.TARGET = target ?: return
        self.withinActionRadius = actionRadiusCheck(self, target)
        for (i in 0 until target.targetedBy.size) {
            if (target.targetedBy[i] == null) {
                target.targetedBy[i] = self
                return
            }
        }
        self.TARGET = null
    }

    fun clearTarget(self: Agent) {
        val target = self.TARGET ?: return
        for (i in 0 until target.targetedBy.size) {
            if (target.targetedBy[i] === self) {
                target.targetedBy[i] = null
                self.TARGET = null
                self.withinActionRadius = false
            }
        }
    }

    fun cheapDistance(a: Agent, b: Entity): Float {
        // Rough Euclidean approximation using voodoo
        val dx = abs(a.x - b.x)
        val dy = abs(a.y - b.y)

        val maxD = max(dx, dy)
        val minD = min(dx, dy)

        return 0.96043384f * maxD + 0.39782473f * minD
    }

    fun cheapDistance(dx: Float, dy: Float): Float {
        // Rough Euclidean approximation using voodoo
        val ax = abs(dx)
        val ay = abs(dy)

        val maxD = max(ax, ay)
        val minD = min(ax, ay)

        return 0.96043384f * maxD + 0.39782473f * minD
    }
}
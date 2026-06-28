package derivative.code.microswarm.entity

import javafx.scene.paint.Color

class Food(
    id: Int,
    x: Float,
    y: Float,
    enabled: Boolean = true
) : Entity(id, x, y, Color.GREEN, enabled) {


    private val MAX_VALUE = 80f
    val INV_MAX_VALUE = 1f / MAX_VALUE
    var value = MAX_VALUE
    var decaySpeed = 0f
    private val REGEN_PER_TICK = 0.5f
    private var previousValue = 100f
    private var depletionPerSecond = 0f
    private var tickerCount = 0f
    var energyPerEat = (value * INV_MAX_VALUE) * 10f
    fun update() {
        val delta = value - previousValue
        depletionPerSecond += delta

        tickerCount++
        if (tickerCount >= 50) {  // sample over 1 second instead of 0.2s
            val rawDecay = ((delta * 50f) / (MAX_VALUE * REGEN_PER_TICK)).coerceIn(-1f, 1f)
            decaySpeed = 0.7f * decaySpeed + 0.3f * rawDecay  // exponential smooth
            tickerCount = 0f
        }
        previousValue = value
        if (value < MAX_VALUE) value += REGEN_PER_TICK
        if (value < 0 && enabled) enabled = false
        if (!enabled && value > 5) enabled = true
        energyPerEat = (value * INV_MAX_VALUE) * 10f
    }

}
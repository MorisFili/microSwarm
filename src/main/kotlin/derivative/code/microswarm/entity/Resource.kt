package derivative.code.microswarm.entity

import javafx.scene.paint.Color

class Resource(
    id: Int,
    x: Float,
    y: Float,
    enabled: Boolean = true
) : Entity(id, x, y, Color.GREEN, enabled) {

    var value = 100f
    var decaySpeed = 0f
    var REGEN_PER_TICK = 1f
    private var REGEN_PER_SECOND = REGEN_PER_TICK * 50f
    private var previousValue = 100f
    private var depletionPerSecond = 0f
    private var tickerCount = 0f
    fun update() {

        val delta = value - previousValue
        depletionPerSecond += delta

        tickerCount++
        if (tickerCount >= 10) {
            decaySpeed = ((depletionPerSecond * 5f) / REGEN_PER_SECOND).coerceIn(-1f, 1f)
            depletionPerSecond = 0f
            tickerCount = 0f
        }
        previousValue = value
        if (value < 100f) value += REGEN_PER_TICK
    }

}
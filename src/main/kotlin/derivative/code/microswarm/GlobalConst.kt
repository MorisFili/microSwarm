package derivative.code.microswarm

import javafx.scene.paint.Color
import kotlin.math.PI
import kotlin.math.acos
import kotlin.math.cos
import kotlin.math.sin

fun managePopHueCounter(hue: Color?, increment: Boolean = true) {
    if (hue == null) return
    when (hue) {
        Color.MAGENTA -> if (increment) Main.magentaAgents.incrementAndGet() else Main.magentaAgents.decrementAndGet()
        Color.WHITE -> if (increment) Main.whiteAgents.incrementAndGet() else Main.whiteAgents.decrementAndGet()
        Color.HOTPINK -> if (increment) Main.pinkAgents.incrementAndGet() else Main.pinkAgents.decrementAndGet()
        Color.ORANGE -> if (increment) Main.orangeAgents.incrementAndGet() else Main.orangeAgents.decrementAndGet()
        Color.SKYBLUE -> if (increment) Main.blueAgents.incrementAndGet() else Main.blueAgents.decrementAndGet()
    }
}

const val TABLE_SIZE = 4096
val sinTable = FloatArray(TABLE_SIZE) { sin((it.toDouble() / TABLE_SIZE - 0.5) * 2.0 * PI).toFloat() }
val cosTable = FloatArray(TABLE_SIZE) { cos((it.toDouble() / TABLE_SIZE - 0.5) * 2.0 * PI).toFloat() }
val acosTable = FloatArray(TABLE_SIZE) { (acos(it.toDouble() / (TABLE_SIZE - 1) * 2.0 - 1.0) / PI).toFloat() }


// Limiting variables
const val PRESENCE_CAP = 50
const val DETECTION_RADIUS = 40f
const val TARGET_RADIUS = DETECTION_RADIUS / 2f
const val ACTION_RADIUS = TARGET_RADIUS / 10f
const val AGENT_DRAW_RADIUS = ACTION_RADIUS * 2
const val DETECTION_RADIUS_SQ = DETECTION_RADIUS * DETECTION_RADIUS
const val GROUP_SIZE = PRESENCE_CAP / 2f

const val MAX_ENERGY = 100f
const val EATING_PROXIMITY = 10f



// Inverse variables
val INV_COUNT = FloatArray(PRESENCE_CAP + 1) { if (it == 0) 0f else 1f / it }
const val INV_DETECTION_RADIUS = 1f / DETECTION_RADIUS
const val INV_TARGET_RADIUS = 1f / TARGET_RADIUS
const val INV_MAX_RADIUS_SQ = 1f / (DETECTION_RADIUS * DETECTION_RADIUS)
const val INV_MAX_ENERGY = 1 / MAX_ENERGY


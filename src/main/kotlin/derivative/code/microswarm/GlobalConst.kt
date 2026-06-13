package derivative.code.microswarm

import javafx.scene.paint.Color
import kotlin.math.PI
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

val TABLE_SIZE = 4096
val sinTable = FloatArray(TABLE_SIZE) { sin((it.toDouble() / TABLE_SIZE - 0.5) * 2.0 * PI).toFloat() }
val cosTable = FloatArray(TABLE_SIZE) { cos((it.toDouble() / TABLE_SIZE - 0.5) * 2.0 * PI).toFloat() }
val PRESENCE_CAP = 50
val DETECTION_RADIUS = 40f
val INV_COUNT = FloatArray(PRESENCE_CAP + 1) { if (it == 0) 0f else 1f / it }
val INV_DETECTION_RADIUS = 1 / DETECTION_RADIUS
val TARGET_RADIUS = DETECTION_RADIUS / 2f
val INV_TARGET_RADIUS = 1 / TARGET_RADIUS
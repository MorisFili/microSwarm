package derivative.code.microswarm

import derivative.code.microswarm.Main.Companion.SELECTED_AGENT_ID
import derivative.code.microswarm.Main.Companion.showAgent
import derivative.code.microswarm.Main.Companion.showIncubating
import derivative.code.microswarm.Main.Companion.showStarving
import derivative.code.microswarm.Simulation.Companion.agents
import derivative.code.microswarm.Simulation.Companion.foods
import javafx.event.EventHandler
import javafx.scene.input.MouseEvent
import javafx.scene.input.ScrollEvent
import javafx.scene.paint.Color

class SimulationRenderer(app: Main) {

    private val app = app
    private val canvas = app.canvas
    private val graphicsContext = app.canvas.graphicsContext2D

    // Viewport
    private var scale = 3.0
    private var viewportX = -(canvas.width * scale) / 2
    private var viewportY = -(canvas.height * scale) / 2

    private var worldLeft = (0.0 - viewportX) / scale
    private var worldTop = (0.0 - viewportY) / scale
    private var worldRight = (canvas.width - viewportX) / scale
    private var worldBottom = (canvas.height - viewportY) / scale


    private var lastMouseX = 0.0
    private var lastMouseY = 0.0


    // Event handlers
    private val zoom = EventHandler<ScrollEvent> { e ->
        val factor = if (e.deltaY > 0) 1.1 else 1.0 / 1.1
        val worldX = (e.x - viewportX) / scale
        val worldY = (e.y - viewportY) / scale
        scale = (scale * factor).coerceIn(1.0, 10.0)
        viewportX = e.x - worldX * scale
        viewportX = viewportX.coerceIn(canvas.width - (canvas.width * scale), 0.0)
        viewportY = e.y - worldY * scale
        viewportY = viewportY.coerceIn(canvas.height - (canvas.height * scale), 0.0)

    }
    private val mouseDrag = EventHandler<MouseEvent> { event: MouseEvent ->
        if (event.isPrimaryButtonDown) {
            viewportX += event.x - lastMouseX
            viewportX = viewportX.coerceIn(canvas.width - (canvas.width * scale), 0.0)
            viewportY += event.y - lastMouseY
            viewportY = viewportY.coerceIn(canvas.height - (canvas.height * scale), 0.0)
        }
        lastMouseX = event.x
        lastMouseY = event.y
    }
    private val mouseMoved = EventHandler<MouseEvent> { event: MouseEvent ->
        lastMouseX = event.x
        lastMouseY = event.y
    }

    init {
        registerHandlers()
    }

    fun render() {
                worldLeft = (0.0 - viewportX) / scale
                worldTop = (0.0 - viewportY) / scale
                worldRight = (canvas.width - viewportX) / scale
                worldBottom = (canvas.height - viewportY) / scale
                clear()
                drawEntities()
    }

    private fun clear() {
        graphicsContext.setTransform(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        graphicsContext.fill = Color.BLACK
        graphicsContext.fillRect(0.0, 0.0, canvas.width, canvas.height)
    }

    private fun drawEntities() {

        graphicsContext.setTransform(scale, 0.0, 0.0, scale, viewportX, viewportY)
        val drawnSize = 1.0 / scale.coerceAtMost(3.0)
        val features = scale >= 3

        for (agent in agents) {
            if (agent == null) continue
            if (!agent.enabled) continue

            // Skip agents outside viewport
            if (agent.x !in worldLeft..worldRight) continue
            if (agent.y !in worldTop..worldBottom) continue

            val cx = agent.x.toDouble()
            val cy = agent.y.toDouble()

            graphicsContext.fill = if (showIncubating) {
                if (agent.INCUBATING) Color.RED else Color.WHITE
            } else if (showStarving) {
                if (agent.energy < 50f) Color.RED else Color.WHITE
            } else if (showAgent) {
                if (agent.id == SELECTED_AGENT_ID) Color.RED else Color.WHITE
            } else agent.hue
            graphicsContext.fillRect(cx - drawnSize, cy - drawnSize,
                drawnSize, drawnSize)

            if (features) {
                val actionIntent = agent.networkAccess().committedActionIntent
                val spatialIntent = agent.networkAccess().committedSpatialIntent

                graphicsContext.fill = intentColor(spatialIntent, 0.02)
                graphicsContext.fillOval(
                    (cx - AGENT_DRAW_RADIUS),
                    (cy - AGENT_DRAW_RADIUS),
                    AGENT_DRAW_RADIUS * 2.0,
                    AGENT_DRAW_RADIUS * 2.0
                )

                graphicsContext.fill = actionColor(actionIntent, 0.1)
                graphicsContext.fillOval(
                    (cx - ACTION_RADIUS),
                    (cy - ACTION_RADIUS),
                    ACTION_RADIUS * 2.0,
                    ACTION_RADIUS * 2.0
                )
            }
        }

        for (resource in foods) {
            val cx = resource.x.toDouble()
            val cy = resource.y.toDouble()
            graphicsContext.fill = if (resource.enabled) resource.hue else Color.RED
            graphicsContext.fillRect(
                cx - drawnSize, cy - drawnSize,
                drawnSize * 2.0, drawnSize * 2.0
            )
            if (features) {
                if (!resource.enabled) continue
                val eatingProximity = EATING_PROXIMITY.toDouble()
                graphicsContext.fill =
                    resource.hue!!.deriveColor(0.0, 1.0, 1.0, 0.1)
                graphicsContext.fillOval(
                    (cx - eatingProximity),
                    (cy - eatingProximity),
                    eatingProximity * 2,
                    eatingProximity * 2
                )
            }
        }
    }

    private fun registerHandlers() {
        canvas.addEventHandler(MouseEvent.MOUSE_DRAGGED, mouseDrag)
        canvas.addEventHandler(MouseEvent.MOUSE_MOVED, mouseMoved)
        canvas.addEventHandler(ScrollEvent.SCROLL, zoom)
    }

    fun removeHandlers() {
        canvas.removeEventHandler(MouseEvent.MOUSE_DRAGGED, mouseDrag)
        canvas.removeEventHandler(MouseEvent.MOUSE_MOVED, mouseMoved)
        canvas.removeEventHandler(ScrollEvent.SCROLL, zoom)
    }


   private fun actionColor(intent: Int, opacity: Double): Color {
       return when (intent) {
           1 -> Color.RED
           2 -> Color.PINK
           3 -> Color.GREEN
           4 -> Color.LIGHTBLUE
           5 -> Color.BROWN
           else -> Color.TRANSPARENT
       }.deriveColor(0.0, 1.0, 1.0, opacity)
    }

    private fun intentColor(intent: Int, opacity: Double): Color {
        return when(intent) {
            1 -> Color.RED
            2 -> Color.GREEN
            3 -> Color.BLUE
            4 -> Color.GRAY
            else -> Color.TRANSPARENT
        }.deriveColor(0.0, 1.0, 1.0, opacity)
    }

}
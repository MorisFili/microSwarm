package derivative.code.microswarm

import derivative.code.microswarm.Simulation.Companion.agents
import javafx.application.Application
import javafx.application.Platform
import javafx.geometry.Insets
import javafx.geometry.Pos
import javafx.scene.Scene
import javafx.scene.canvas.Canvas
import javafx.scene.control.Button
import javafx.scene.control.CheckBox
import javafx.scene.control.ComboBox
import javafx.scene.control.Label
import javafx.scene.control.RadioButton
import javafx.scene.control.ScrollPane
import javafx.scene.control.ToggleGroup
import javafx.scene.input.Clipboard
import javafx.scene.input.ClipboardContent
import javafx.scene.layout.Background
import javafx.scene.layout.BackgroundFill
import javafx.scene.layout.Border
import javafx.scene.layout.BorderStroke
import javafx.scene.layout.BorderStrokeStyle
import javafx.scene.layout.BorderWidths
import javafx.scene.layout.CornerRadii
import javafx.scene.layout.HBox
import javafx.scene.layout.Priority
import javafx.scene.layout.Region
import javafx.scene.layout.StackPane
import javafx.scene.layout.VBox
import javafx.scene.paint.Color
import javafx.scene.text.Font
import javafx.scene.text.FontWeight
import javafx.stage.Screen
import javafx.stage.Stage
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.system.exitProcess

class Main : Application() {

    // Field variables
    companion object {
        var CANVAS_X = 1000.0
        var CANVAS_Y = 1000.0

        // Panel statistics counters
        @Volatile
        var AVG_SIM_TICK_TIME = 0

        @Volatile
        var MAX_SIM_TICK_TIME = 0

        @Volatile
        var TICKS_OVER_12 = 0

        @Volatile
        var TICKS_OVER_16 = 0

        @Volatile
        var TICKS_OVER_20 = 0

        @Volatile
        var SELECTED_AGENT_ID = 0

        @Volatile
        var MALE_POP = 0

        @Volatile
        var RENEGADE_POP = 0

        @Volatile
        var TOP_RENEGADE_ID = -1

        @Volatile
        var PARAGON_POP = 0

        @Volatile
        var TOP_PARAGON_ID = -1

        @Volatile
        var showIncubating = false

        @Volatile
        var showStarving = false

        @Volatile
        var showAgent = false

        @Volatile
        var showAllWeights = false


        val populationCounter = AtomicInteger(0)
        val killedOwnHue = AtomicInteger(0)
        val killedOwnHueByStealing = AtomicInteger(0)
        val killedOtherHue = AtomicInteger(0)
        val killedOtherHueByStealing = AtomicInteger(0)
        val sharedWithSameHue = AtomicInteger(0)
        val sharedWithOtherHue = AtomicInteger(0)
        val stoleFromSameHue = AtomicInteger(0)
        val stoleFromOtherHue = AtomicInteger(0)
        val spawnedWithOwnHue = AtomicInteger(0)
        val spawnedWithOtherHue = AtomicInteger(0)
        val magentaAgents = AtomicInteger(0)
        val whiteAgents = AtomicInteger(0)
        val pinkAgents = AtomicInteger(0)
        val orangeAgents = AtomicInteger(0)
        val blueAgents = AtomicInteger(0)
        val starvationCounter = AtomicInteger(0)
    }

    // Class properties
    val canvas = Canvas(CANVAS_X, CANVAS_Y)
    val renderer = SimulationRenderer(this)
    val topLabel = Label("Agent ID: 0").apply {
        textFill = Color.rgb(220, 225, 235)
        font = Font.font("System", FontWeight.SEMI_BOLD, 14.0)
    }
    val selector = ComboBox<Int>().apply {
        isEditable = false
        promptText = "Select Agent"
        maxWidth = Double.MAX_VALUE
        padding = Insets(0.0, 0.0, 0.0, 5.0)
    }
    val outputText = Label().apply {
        textFill = Color.rgb(220, 225, 235)
        padding = Insets(10.0)
        isWrapText = true
        maxWidth = Double.MAX_VALUE
        background = Background(
            BackgroundFill(
                Color.rgb(24, 26, 30),
                CornerRadii.EMPTY,
                Insets.EMPTY
            )
        )
    }

    @Volatile
    var RUNNING = false

    val simLoop: ScheduledExecutorService = Executors.newSingleThreadScheduledExecutor { runnable ->
        Thread(runnable).apply { isDaemon = true }
    }

    val simulation = Simulation(this)

    override fun start(stage: Stage) {

        val topBar = HBox()
        topBar.apply {

            padding = Insets(10.0)
            spacing = 10.0
            alignment = Pos.CENTER_LEFT

            background = Background(
                BackgroundFill(
                    Color.rgb(24, 26, 30),
                    CornerRadii.EMPTY,
                    Insets.EMPTY
                )
            )

            border = Border(
                BorderStroke(
                    Color.rgb(55, 58, 65),
                    BorderStrokeStyle.SOLID,
                    CornerRadii.EMPTY,
                    BorderWidths(0.0, 0.0, 1.0, 0.0)
                )
            )
        }

        val startBtn = Button() // Start/Pause/Resume btn
        startBtn.text = "Start"

        val resetBtn = Button()
        resetBtn.text = "Reset"
        resetBtn.isDisable = true

        val saveBtn = Button()
        saveBtn.text = "Save"
        saveBtn.isDisable = true

        val loadBtn = Button()
        loadBtn.text = "Load"
        loadBtn.isDisable = true

        val topBarSpacer = Region()
        HBox.setHgrow(topBarSpacer, Priority.ALWAYS)

        val multiThreadToggle = CheckBox("Multi-thread: Enabled").apply {
            textFill = Color.rgb(220, 225, 235)
            isSelected = true
            selectedProperty().addListener { _, _, selected ->
                if (!RUNNING) {
                    simulation.multiThreaded = selected
                    text = if (selected) "Multi-thread: Enabled" else "Multi-thread: Disabled"
                }
            }
        }


        startBtn.setOnAction {
            RUNNING = !RUNNING
            startBtn.text = if (RUNNING) "Pause" else "Resume"
            multiThreadToggle.isDisable = RUNNING
        }

        topBar.children.addAll(startBtn, resetBtn, saveBtn, loadBtn, topBarSpacer, multiThreadToggle)


        canvas.setOnMouseClicked { e ->
            simulation.selectAgent(e.x, e.y)
            topLabel.text = "Agent ID: $SELECTED_AGENT_ID"
        }

        val canvasPane = StackPane(canvas).apply {
            minWidth = CANVAS_X
            prefWidth = CANVAS_X
            maxWidth = CANVAS_X
            minHeight = CANVAS_Y
            prefHeight = CANVAS_Y
            maxHeight = CANVAS_Y
        }

        // ========================== AGENT INFO SECTION ==========================

        val rightPanel = VBox()
        rightPanel.apply {
            padding = Insets(0.0, 0.0, 0.0, 5.0)
            spacing = 10.0

            background = Background(
                BackgroundFill(
                    Color.rgb(24, 26, 30),
                    CornerRadii.EMPTY,
                    Insets.EMPTY
                )
            )

            border = Border(
                BorderStroke(
                    Color.rgb(55, 58, 65),
                    BorderStrokeStyle.SOLID,
                    CornerRadii.EMPTY,
                    BorderWidths(0.0, 0.0, 0.0, 2.0)
                )
            )
        }
        val copyBtn = Button("Copy Agent Dump").apply {
            setOnAction {
                Clipboard.getSystemClipboard().setContent(
                    ClipboardContent()
                        .apply { putString(outputText.text) }
                )
            }
        }

        val spacer = Region()
        HBox.setHgrow(spacer, Priority.ALWAYS)

        val detailedWeightsToggle = CheckBox("Detailed View: Disabled").apply {
            textFill = Color.rgb(220, 225, 235)
            isSelected = false
            font = Font.font("System", FontWeight.SEMI_BOLD, 14.0)
            padding = Insets(0.0, 0.0, 0.0, 10.0) // top, right, bottom, left
            selectedProperty().addListener { _, _, selected ->
                showAllWeights = selected
                text = if (selected) "Detailed View: Enabled" else "Detailed View: Disabled"

            }
        }


        val topWrapper = HBox(topLabel, detailedWeightsToggle, spacer, copyBtn).apply {
            maxWidth = Double.MAX_VALUE
            padding = Insets(0.0, 0.0, 0.0, 5.0)
        }

        val innerPanel = HBox().apply {
            padding = Insets(0.0, 0.0, 0.0, 5.0)
            maxWidth = Double.MAX_VALUE
            prefHeight = 700.0
            maxHeight = 700.0
        }

        val scrollableWrapper = ScrollPane(outputText).apply {
            isFitToWidth = true
            hbarPolicy = ScrollPane.ScrollBarPolicy.NEVER
            vbarPolicy = ScrollPane.ScrollBarPolicy.AS_NEEDED

            maxWidthProperty().bind(innerPanel.widthProperty().multiply(0.5))
            maxHeight = 700.0
            prefHeight = 700.0

        }

        val instrumentPanel = VBox().apply {
            maxWidthProperty().bind(innerPanel.widthProperty().multiply(0.5))
            //padding = Insets(0.0, 0.0, 0.0, 0.0) // top, right, bottom, left
            prefHeight = 700.0
            maxHeight = 700.0
        }

        val highLightGroup = ToggleGroup()

        val highlightNone = RadioButton("None").apply {
            textFill = Color.rgb(220, 225, 235)
            padding = Insets(10.0)
            toggleGroup = highLightGroup
            isSelected = true  // default
        }

        val highlightHungry = RadioButton("Highlight Hungry").apply {
            textFill = Color.rgb(220, 225, 235)
            padding = Insets(10.0)
            toggleGroup = highLightGroup
        }

        val highlightIncubating = RadioButton("Highlight Incubating").apply {
            textFill = Color.rgb(220, 225, 235)
            padding = Insets(10.0)
            toggleGroup = highLightGroup
        }

        val highlightAgent = RadioButton("Highlight Selected Agent").apply {
            textFill = Color.rgb(220, 225, 235)
            padding = Insets(10.0)
            toggleGroup = highLightGroup
        }

        highLightGroup.selectedToggleProperty().addListener { _, _, newToggle ->
            when (newToggle) {
                highlightHungry -> {
                    showStarving = true
                    showIncubating = false
                    showAgent = false
                }

                highlightIncubating -> {
                    showStarving = false
                    showIncubating = true
                    showAgent = false
                }

                highlightAgent -> {
                    showStarving = false
                    showIncubating = false
                    showAgent = true
                }

                else -> {
                    showStarving = false
                    showIncubating = false
                    showAgent = false
                }
            }
        }



        instrumentPanel.children.addAll(
            highlightNone, highlightHungry, highlightIncubating,
            highlightAgent
        )

        HBox.setHgrow(scrollableWrapper, Priority.ALWAYS)
        HBox.setHgrow(instrumentPanel, Priority.ALWAYS)

        innerPanel.children.addAll(scrollableWrapper, instrumentPanel)

        rightPanel.children.addAll(
            topBar,
            topWrapper,
            selector,
            innerPanel
        )

        // ------------------------------------------------------------------------

        val bottomArea = HBox()
        bottomArea.apply {
            spacing = 15.0
            padding = Insets(12.0, 10.0, 12.0, 10.0)

            background = Background(
                BackgroundFill(
                    Color.rgb(24, 26, 30),
                    CornerRadii.EMPTY,
                    Insets.EMPTY
                )
            )

            border = Border(
                BorderStroke(
                    Color.rgb(55, 58, 65),
                    BorderStrokeStyle.SOLID,
                    CornerRadii.EMPTY,
                    BorderWidths(1.0, 0.0, 0.0, 0.0)
                )
            )

        }

        val populationLabel = Label().apply {
            textFill = Color.rgb(220, 225, 235)
        }
        val agentColorLabel = Label().apply {
            textFill = Color.rgb(220, 225, 235)
        }
        val deathLabel = Label().apply {
            textFill = Color.rgb(220, 225, 235)
        }
        val ecoTransferLabel = Label().apply {
            textFill = Color.rgb(220, 225, 235)
        }
        val spawnLabel = Label().apply {
            textFill = Color.rgb(220, 225, 235)
        }
        val systemInfoLabel = Label().apply {
            textFill = Color.rgb(220, 225, 235)
        }

        val space = Region()
        HBox.setHgrow(space, Priority.ALWAYS)

        bottomArea.children.addAll(
            populationLabel, agentColorLabel,
            deathLabel, ecoTransferLabel, spawnLabel, systemInfoLabel
        )

        rightPanel.children.add(bottomArea)

        val mainArea = HBox(canvasPane, rightPanel).apply {
            padding = Insets(5.0)
            spacing = 5.0
            background = Background(
                BackgroundFill(
                    Color.rgb(24, 26, 30),
                    CornerRadii.EMPTY,
                    Insets.EMPTY
                )
            )
        }
        val scene = Scene(mainArea)

        HBox.setHgrow(rightPanel, Priority.ALWAYS)

        // =================== EVENTS BLOCK ===================
        selector.setOnAction {
            if (selector.value == null) return@setOnAction
            SELECTED_AGENT_ID = selector.value
            topLabel.text = "Agent ID: $SELECTED_AGENT_ID"
        }

        // -----------------------------------------------------
        // Main simulation thread at 50hz
        val simulationLoop = object : Runnable {
            var tickCount = 0
            var totalUpdateNs = 0L
            var maxUpdateNs = 0L
            var over12ms = 0
            var over16ms = 0
            var over20ms = 0
            var delay = 10L
            override fun run() {
                try {
                    if (RUNNING) {
                        val start = System.nanoTime()

                        simulation.update()
                        Platform.runLater { renderer.render() }

                        val elapsed = System.nanoTime() - start
                        val elapsedMs = elapsed / 1_000_000L
                        delay = (20 - elapsedMs).coerceAtLeast(0)


                        totalUpdateNs += elapsed
                        if (elapsed > maxUpdateNs) maxUpdateNs = elapsed
                        tickCount++
                        if (elapsed > 12_000_000L) over12ms++
                        if (elapsed > 16_000_000L) over16ms++
                        if (elapsed > 20_000_000L) over20ms++
                        if (tickCount >= 100) {
                            AVG_SIM_TICK_TIME = (totalUpdateNs / tickCount / 1000L).toInt()
                            MAX_SIM_TICK_TIME = (maxUpdateNs / 1000L).toInt()
                            TICKS_OVER_12 = over12ms
                            TICKS_OVER_16 = over16ms
                            TICKS_OVER_20 = over20ms

                            tickCount = 0
                            totalUpdateNs = 0L
                            maxUpdateNs = 0L
                            over12ms = 0
                            over16ms = 0
                            over20ms = 0
                        }
                    } else {
                        Platform.runLater { renderer.render() }
                    }
                } catch (t: Throwable) {
                    println(t.stackTraceToString())
                } finally {
                    if (!simLoop.isShutdown) simLoop.schedule(this, delay, TimeUnit.MILLISECONDS)                }
            }
        }
        simLoop.schedule(simulationLoop, 0, TimeUnit.MILLISECONDS)

        // Main sim thread running analysis 2.5 times per sec
        simLoop.scheduleAtFixedRate(analyze@{
            try {
                if (RUNNING) {
                    val text = agents.firstOrNull { it?.id == SELECTED_AGENT_ID }
                        ?.analysis()
                        ?: return@analyze
                    Platform.runLater {
                        outputText.text = text
                        val popCounter = populationCounter.get()
                        populationLabel.text =
                            "Population: ${popCounter}\n" +
                                    "Male pop: $MALE_POP \n" +
                                    "Renegade pop: $RENEGADE_POP \n" +
                                    "Top Renegade: $TOP_RENEGADE_ID \n" +
                                    "Paragon pop: $PARAGON_POP \n" +
                                    "Top Paragon: $TOP_PARAGON_ID \n"
                        agentColorLabel.text =
                            "Population Hues\n" +
                                    "Magenta: %.2f%%\n".format(
                                        magentaAgents.get().toDouble() /
                                                popCounter * 100.0
                                    ) +
                                    "White: %.2f%%\n".format(
                                        whiteAgents.get().toDouble() /
                                                popCounter * 100.0
                                    ) +
                                    "Pink: %.2f%%\n".format(
                                        pinkAgents.get().toDouble() /
                                                popCounter * 100.0
                                    ) +
                                    "Orange: %.2f%%\n".format(
                                        orangeAgents.get().toDouble() /
                                                popCounter * 100.0
                                    ) +
                                    "Blue: %.2f%%\n".format(
                                        blueAgents.get().toDouble() /
                                                popCounter * 100.0
                                    )
                        deathLabel.text =
                            "Killed Other Hue: ${killedOtherHue.get()}\n" +
                                    "Killed Other Hue w Stealing: ${killedOtherHueByStealing.get()}\n" +
                                    "Killed Own Hue: ${killedOwnHue.get()}\n" +
                                    "Killed Own Hue w Stealing: ${killedOwnHueByStealing.get()}\n" +
                                    "Starvation: ${starvationCounter.get()}"
                        ecoTransferLabel.text =
                            "Stole fr Other Hue: ${stoleFromOtherHue.get()}\n" +
                                    "Stole fr Same Hue: ${stoleFromSameHue.get()}\n" +
                                    "Shared w Other Hue: ${sharedWithOtherHue.get()}\n" +
                                    "Shared w Same Hue: ${sharedWithSameHue.get()}"
                        spawnLabel.text =
                            "Spawned w Other Hue: ${spawnedWithOtherHue.get()}\n" +
                                    "Spawned w Same Hue: ${spawnedWithOwnHue.get()}\n"
                        systemInfoLabel.text =
                            "--- Over 100 loops: --- \n" +
                                    "Avg. Loop Time: ${AVG_SIM_TICK_TIME / 1000f}ms \n" +
                                    "Max Loop Time: ${MAX_SIM_TICK_TIME / 1000f}ms \n" +
                                    "Loops over 12ms: $TICKS_OVER_12 \n" +
                                    "Loops over 16ms: $TICKS_OVER_16 \n" +
                                    "Loops over 20ms: $TICKS_OVER_20"
                    }
                }
            } catch (t: Throwable) {
                println(t.stackTraceToString())
            }
        }, 0, 400, TimeUnit.MILLISECONDS)

        stage.scene = scene
        stage.width = Screen.getPrimary().visualBounds.width
        stage.height = Screen.getPrimary().visualBounds.height
        stage.isResizable = false
        stage.title = "Simulation"
        stage.show()
    }

    override fun stop() {
        RUNNING = false
        renderer.removeHandlers()
        simLoop.shutdownNow()
        simulation.shutDownThreads()
        Platform.exit()
        exitProcess(0)
    }

}
  

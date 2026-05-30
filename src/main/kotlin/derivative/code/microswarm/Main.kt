package derivative.code.microswarm

import derivative.code.microswarm.Simulation.Companion.agents
import derivative.code.microswarm.Simulation.Companion.resources
import javafx.animation.AnimationTimer
import javafx.application.Application
import javafx.application.Platform
import javafx.beans.binding.Bindings
import javafx.geometry.Insets
import javafx.geometry.Pos
import javafx.scene.Scene
import javafx.scene.canvas.Canvas
import javafx.scene.control.Button
import javafx.scene.control.CheckBox
import javafx.scene.control.ComboBox
import javafx.scene.control.Label
import javafx.scene.control.ScrollPane
import javafx.scene.control.Slider
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
import javafx.scene.layout.Pane
import javafx.scene.layout.Priority
import javafx.scene.layout.Region
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
        var GLOBAL_AVG_CREDIT = 0

        @Volatile
        var MALE_POP = 0

        @Volatile
        var APATHIC_POP = 0

        @Volatile
        var RENEGADE_POP = 0

        @Volatile
        var VAL_MULTIPL = 1f
        @Volatile
        var LR_MULTIPL = 1f

        val populationCounter = AtomicInteger(0)
        val killedOwnHue = AtomicInteger(0)
        val killedOtherHue = AtomicInteger(0)
        val spawnedWithOwnHue = AtomicInteger(0)
        val spawnedWithOtherHue = AtomicInteger(0)
        val spawnAttempts = AtomicInteger(0)
        val killAttempts = AtomicInteger(0)
        val magentaAgents = AtomicInteger(0)
        val whiteAgents = AtomicInteger(0)
        val pinkAgents = AtomicInteger(0)
        val orangeAgents = AtomicInteger(0)
        val blueAgents = AtomicInteger(0)
    }

    // Class properties
    val topLabel = Label("Agent ID: 0").apply {
        textFill = Color.rgb(220, 225, 235)
        font = Font.font("System", FontWeight.SEMI_BOLD, 14.0)
    }
    val selector = ComboBox<Int>().apply {
        isEditable = false
        promptText = "Select Agent"
        maxWidth = Double.MAX_VALUE
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
    private var RUNNING = false

    val simLoop: ScheduledExecutorService = Executors.newSingleThreadScheduledExecutor { runnable ->
        Thread(runnable).apply { isDaemon = true }
    }

    val simulation = Simulation(this)

    override fun start(stage: Stage) {

        val topBar = HBox()
        topBar.apply {
            prefHeight = 48.0
            minHeight = 48.0
            maxHeight = 48.0

            spacing = 12.0
            alignment = Pos.CENTER_LEFT
            padding = Insets(8.0, 14.0, 8.0, 14.0)

            background = Background(
                BackgroundFill(
                    Color.rgb(28, 30, 34),
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

        val canvas = Canvas(CANVAS_X, CANVAS_Y)
        val graphicsContext = canvas.graphicsContext2D
        val canvasPane = Pane(canvas)
        canvasPane.apply {
            canvasPane.minWidth = CANVAS_X
            canvasPane.prefWidth = CANVAS_X
            canvasPane.maxWidth = CANVAS_X

            canvasPane.minHeight = CANVAS_Y
            canvasPane.prefHeight = CANVAS_Y
            canvasPane.maxHeight = CANVAS_Y
        }

        // ========================== AGENT INFO SECTION ==========================

        val rightPanel = VBox()
        rightPanel.apply {
            spacing = 12.0
            padding = Insets(14.0)

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
                    BorderWidths(0.0, 0.0, 0.0, 1.0)
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

        val topWrapper = HBox(topLabel, spacer, copyBtn).apply {
            maxWidth = Double.MAX_VALUE
        }

        val innerPanel = HBox().apply {
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
            prefHeight = 700.0
            maxHeight = 700.0
        }

        val learningAmplifierSlider = Slider().apply {
            min = 1.0
            max = 10.0
            value = 1.0

            valueProperty().addListener { _, _, newVal ->
                VAL_MULTIPL = newVal.toFloat()
            }
        }
        val learningAmplifierLabel = Label().apply {
            textFill = Color.rgb(220, 225, 235)
            textProperty().bind(
                Bindings.createStringBinding(
                    {
                        "Valence Multiplier: %.3f".format(VAL_MULTIPL)
                    },
                    learningAmplifierSlider.valueProperty()
                )
            )
        }
        val valenceAmpBox = VBox(learningAmplifierLabel, learningAmplifierSlider).apply {
            spacing = 6.0
            padding = Insets(12.0)
        }

        val learningRateSlider = Slider().apply {
            min = 1.0
            max = 10.0
            value = 1.0

            valueProperty().addListener { _, _, newVal ->
                LR_MULTIPL = newVal.toFloat()
            }
        }
        val learningRateLabel = Label().apply {
            textFill = Color.rgb(220, 225, 235)
            textProperty().bind(
                Bindings.createStringBinding(
                    {
                        "Learning rate: %.3f".format(LR_MULTIPL)
                    },
                    learningRateSlider.valueProperty()
                )
            )
        }
        val learningRateBox = VBox(learningRateLabel, learningRateSlider).apply {
            spacing = 6.0
            padding = Insets(12.0)
        }

        instrumentPanel.children.addAll(valenceAmpBox, learningRateBox)

        HBox.setHgrow(scrollableWrapper, Priority.ALWAYS)
        HBox.setHgrow(instrumentPanel, Priority.ALWAYS)

        innerPanel.children.addAll(scrollableWrapper, instrumentPanel)

        rightPanel.children.addAll(
            topWrapper,
            selector,
            innerPanel
        )

        // ------------------------------------------------------------------------

        val bottomArea = HBox()
        bottomArea.apply {
            spacing = 12.0
            padding = Insets(12.0, 14.0, 12.0, 14.0)

            background = Background(
                BackgroundFill(
                    Color.rgb(22, 24, 28),
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
        val killedLabel = Label().apply {
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

        bottomArea.children.addAll(populationLabel, agentColorLabel,
            killedLabel, spawnLabel, space, systemInfoLabel)

        rightPanel.children.add(bottomArea)

        val mainArea = HBox(canvasPane, rightPanel)

        val root = VBox(topBar, mainArea)
        val scene = Scene(root)

        VBox.setVgrow(mainArea, Priority.ALWAYS)
        HBox.setHgrow(rightPanel, Priority.ALWAYS)


        // =================== EVENTS BLOCK ===================
        selector.setOnAction {
            if (selector.value == null) return@setOnAction
            SELECTED_AGENT_ID = selector.value
            topLabel.text = "Agent ID: $SELECTED_AGENT_ID"
        }

        // -----------------------------------------------------
        // Main simulation thread at 50hz
        var tickCount = 0
        var totalUpdateNs = 0L
        var maxUpdateNs = 0L
        var over12ms = 0
        var over16ms = 0
        var over20ms = 0
        simLoop.scheduleAtFixedRate({
            try {
                if (RUNNING) {
                    val start = System.nanoTime()
                    simulation.update()
                    val elapsed = System.nanoTime() - start
                    totalUpdateNs += elapsed
                    if (elapsed > maxUpdateNs) maxUpdateNs = elapsed
                    tickCount++
                    if (elapsed > 12_000_000L) over12ms++
                    if (elapsed > 16_000_000L) over16ms++
                    if (elapsed > 20_000_000L) over20ms++
                    if (tickCount >= 100) {
                        AVG_SIM_TICK_TIME = (totalUpdateNs / tickCount / 1000).toInt()
                        MAX_SIM_TICK_TIME = (maxUpdateNs / 1000).toInt()
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
                }
            } catch (t: Throwable) {
                println(t.stackTraceToString())
            }
        }, 0, 20, TimeUnit.MILLISECONDS)

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
                                    "Apathic pop: $APATHIC_POP \n" +
                                    "Renegade pop: $RENEGADE_POP \n" +
                                    "Global Avg. Credit: $GLOBAL_AVG_CREDIT"
                        agentColorLabel.text =
                            "Population Hues\n" +
                                    "Magenta: %.2f%%\n".format(magentaAgents.get().toDouble() /
                                            popCounter * 100.0) +
                                    "White: %.2f%%\n".format(whiteAgents.get().toDouble() /
                                            popCounter * 100.0) +
                                    "Pink: %.2f%%\n".format(pinkAgents.get().toDouble() /
                                            popCounter * 100.0) +
                                    "Orange: %.2f%%\n".format(orangeAgents.get().toDouble() /
                                            popCounter * 100.0) +
                                    "Blue: %.2f%%\n".format(blueAgents.get().toDouble() /
                                            popCounter * 100.0)
                        killedLabel.text =
                            "Killed Other Hue: ${killedOtherHue.get()}\n" +
                                    "Killed Same Hue: ${killedOwnHue.get()}\n" +
                                    "Kill Attempts: ${killAttempts.get()}"
                        spawnLabel.text =
                            "Spawned Other Hue: ${spawnedWithOtherHue.get()}\n" +
                                    "Spawned Same Hue: ${spawnedWithOwnHue.get()}\n" +
                                    "Spawn Attempts: ${spawnAttempts.get()}"
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

            object : AnimationTimer() {
                override fun handle(now: Long) {
                    if (RUNNING) {
                        graphicsContext.fill = Color.BLACK
                        graphicsContext.fillRect(0.0, 0.0, canvas.width, canvas.height)

                        for (resource in resources) {
                            if (!resource.enabled) continue
                            graphicsContext.fill = resource.hue ?: Color.WHITE
                            graphicsContext.fillRect(resource.x.toDouble(), resource.y.toDouble(), 2.0, 2.0)
                        }

                        for (entity in agents) {
                            if (entity == null) continue
                            if (!entity.enabled) continue
                            graphicsContext.fill = entity.hue ?: Color.WHITE
                            graphicsContext.fillRect(entity.x.toDouble(), entity.y.toDouble(), 1.0, 1.0)
                        }
                    }
                }
            }.start()

        stage.scene = scene
        stage.width = Screen.getPrimary().visualBounds.width
        stage.height = Screen.getPrimary().visualBounds.height
        stage.isResizable = false
        stage.title = "Simulation"
        stage.show()
    }

    override fun stop() {
        RUNNING = false
        simLoop.shutdownNow()
        simulation.shutDownThreads()
        Platform.exit()
        exitProcess(0)
    }

}
  

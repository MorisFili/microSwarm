package derivative.code.microswarm


import derivative.code.microswarm.Main.Companion.CANVAS_X
import derivative.code.microswarm.Main.Companion.CANVAS_Y
import derivative.code.microswarm.entity.Agent
import derivative.code.microswarm.entity.Entity
import derivative.code.microswarm.entity.Food
import derivative.code.microswarm.entity.GeneticMaterial
import derivative.code.microswarm.network.Network
import javafx.application.Platform
import javafx.scene.paint.Color
import java.util.*
import java.util.concurrent.Executors
import java.util.concurrent.Future


class Simulation(
    private val application: Main
) {

    // Grid Settings
    companion object {
        val rng = Random()
        val MAX_ENTITY_COUNT = 5000
        val INITIAL_ENTITY_COUNT = 1000

        val palette = arrayOf( // Reserved: green = resource fields, red = renegades
            Color.MAGENTA,
            Color.WHITE,
            Color.HOTPINK,
            Color.ORANGE,
            Color.SKYBLUE
        )
        const val CELL_SIZE = 50
        const val CELLS_PER_ROW = 1000 / CELL_SIZE
        const val CELLS_PER_COLUMN = 1000 / CELL_SIZE
        const val MAX_PER_CELL = 2500 // maximum, tweak later
        const val NET_IN = 37
        val gridCellCount = Array(CELLS_PER_ROW) { IntArray(CELLS_PER_COLUMN) }
        val entityGrid = Array(CELLS_PER_ROW) {
            Array(CELLS_PER_COLUMN) {
                arrayOfNulls<Entity>(MAX_PER_CELL)
            }
        }
        val occupancyGrid = Array(CANVAS_X.toInt()) { BooleanArray(CANVAS_Y.toInt()) }
        var nextAgentIndex = 0
        val agents = Array(MAX_ENTITY_COUNT) { i ->
            if (i < INITIAL_ENTITY_COUNT) {
                val x = rng.nextFloat(0f, CANVAS_X.toFloat())
                val y = rng.nextFloat(0f, CANVAS_Y.toFloat())
                val hue = palette[rng.nextInt(palette.size)]
                val nn = Network(NET_IN, 2, 3, 3)
                nextAgentIndex++
                Agent(x, y, i, hue, network = nn)
            } else null
        }
        val foods = Array(50) { i ->
            val x = rng.nextFloat(50f,950f)
            val y = rng.nextFloat(50f,950f)
            Food(i, x, y)
        }
    }

    val activeAgents = arrayOfNulls<Agent>(MAX_ENTITY_COUNT)
    var multiThreaded = true
    val workerCount = Runtime.getRuntime().availableProcessors()
    private val threads = Executors.newFixedThreadPool(workerCount)
    private val futures = mutableListOf<Future<*>>()

    fun shutDownThreads() {
        threads.shutdownNow()
    }

    var triggerCounter = 0

    init {

        for (resource in foods) {
            occupancyGrid[resource.x.toInt()][resource.y.toInt()] = true
        }

        // Fill occupancy grid with spawn coordinates
        for (entity in agents) {
            if (entity == null) continue
            managePopHueCounter(entity.hue)
            occupancyGrid[entity.x.toInt()][entity.y.toInt()] = true
        }
        Main.populationCounter.set(INITIAL_ENTITY_COUNT)
        Platform.runLater { application.selector.items.addAll(agents.indices) }
    }

    fun update() {
        triggerCounter++
        gridUpdate()

        if (multiThreaded) {

            // Fill active agent array
            var activeAgentsCount = 0
            activeAgents.fill(null)
            for (agent in agents) {
                if (agent == null) continue
                if (!agent.enabled) continue
                activeAgents[activeAgentsCount] = agent
                activeAgentsCount++
            }

            if (activeAgentsCount == 0) { // All dead
                application.RUNNING = false
                return
            }

            val chunkSize = (activeAgentsCount + workerCount - 1) / workerCount

            var index = 0
            futures.clear()

            // Sync phase
            for (i in 0 until activeAgentsCount) {
                activeAgents[i]!!.syncGenerateIntent()
            }

            // Async phase
            while (index < activeAgentsCount) {
                val from = index
                val to = minOf(from + chunkSize, activeAgentsCount)
                futures += threads.submit {
                    var i = from
                    while (i < to) {
                        activeAgents[i]!!.asyncFeedForward()
                        i++
                    }
                }
                index = to
            }

            for (future in futures) future.get()

            // Sync phase
            for (i in 0 until activeAgentsCount) {
                val agent = activeAgents[i]!!
                agent.syncPerformAction()
                if (agent.MATE_CONDITION) {
                    if (Main.populationCounter.get() <= MAX_ENTITY_COUNT * 99 / 100) {
                        reproduce(agent, agent.INCUBATION_MATERIAL!!)
                    }

                    agent.MATE_CONDITION = false
                    agent.clearTarget()
                }
            }

            // Async phase
            index = 0
            futures.clear()
            while (index < activeAgentsCount) {
                val from = index
                val to = minOf(from + chunkSize, activeAgentsCount)
                futures += threads.submit {
                    var i = from
                    while (i < to) {
                        activeAgents[i]!!.asyncActionEvaluation()
                        i++
                    }
                }
                index = to
            }

            for (future in futures) future.get()

            // Second sync phase
            for (i in 0 until activeAgentsCount) {
                activeAgents[i]!!.syncMovement()
            }

            // Last phase (async)
            index = 0
            futures.clear()
            while (index < activeAgentsCount) {
                val from = index
                val to = minOf(from + chunkSize, activeAgentsCount)
                futures += threads.submit {
                    var i = from
                    while (i < to) {
                        activeAgents[i]!!.asyncStateEvaluation()
                        i++
                    }
                }
                index = to
            }

            for (future in futures) future.get()

        } else {
            // Single thread loop
            for (entity in agents) {
                if (entity == null) continue
                if (!entity.enabled) continue
                entity.syncGenerateIntent()
                entity.asyncFeedForward()
                entity.syncPerformAction()
                if (entity.MATE_CONDITION) {
                    if (entity.TARGET != null) {
                        if (Main.populationCounter.get() <= MAX_ENTITY_COUNT * 99 / 100) {
                            reproduce(entity, entity.INCUBATION_MATERIAL!!)
                        }
                    }
                    entity.MATE_CONDITION = false
                    entity.clearTarget()
                }
                entity.asyncActionEvaluation()
                entity.syncMovement()
                entity.asyncStateEvaluation()
            }
        }

        for (resource in foods) {
            resource.update()
            if (resource.value <= 0) {
                val coordinates = getNearestUnoccupiedCoordinate(
                    rng.nextFloat(50f,950f),
                    rng.nextFloat(50f,950f)
                )
                resource.x = coordinates[0].toFloat()
                resource.y = coordinates[1].toFloat()
                resource.value = resource.MAX_VALUE
            }
        }

        if (triggerCounter >= 25) {
            var totalMale = 0
            var totalApathic = 0
            var totalRene = 0
            for (entity in agents) {
                if (entity == null) continue
                if (!entity.enabled) continue
                if (entity.isMale) totalMale++
                if (entity.isApathic) totalApathic++
                if (entity.REPUTATION < 0.5f) totalRene++
            }
            Main.MALE_POP = totalMale
            Main.APATHIC_POP = totalApathic
            Main.RENEGADE_POP = totalRene
            triggerCounter = 0
        }

    }

    private fun reproduce(a1: Agent, a2: GeneticMaterial) {
        val a1weights = a1.exportWeights()
        val a2weights = a2.weights

        val input = a1weights.transferInput
        val input2 = a2weights.transferInput
        for (i in 0 until input.size) {
            if (rng.nextFloat() < 0.5f) {
                for (j in input[i].indices) input[i][j] = input2[i][j]
            }
        }

        val intent = a1weights.transferIntent
        val intent2 = a2weights.transferIntent
        for (i in 0 until intent.size) {
            if (rng.nextFloat() < 0.5f) {
                for (j in intent[i].indices) intent[i][j] = intent2[i][j]
            }
        }

        val adrenal = a1weights.transferAdrenergic
        val adrenal2 = a2weights.transferAdrenergic
        for (i in 0 until adrenal.size) {
            if (rng.nextFloat() < 0.5f) {
                for (j in adrenal[i].indices) adrenal[i][j] = adrenal2[i][j]
            }
        }

        val dopamine = a1weights.transferDopaminergic
        val dopamine2 = a2weights.transferDopaminergic
        for (i in 0 until dopamine.size) {
            if (rng.nextFloat() < 0.5f) {
                for (j in dopamine[i].indices) dopamine[i][j] = dopamine2[i][j]
            }
        }

        val sero = a1weights.transferSerotonergic
        val sero2 = a2weights.transferSerotonergic
        for (i in 0 until sero.size) {
            if (rng.nextFloat() < 0.5f) {
                for (j in sero[i].indices) sero[i][j] = sero2[i][j]
            }
        }

        val spawnCoordinates = getNearestUnoccupiedCoordinate(a1.x, a1.y)
        val x = spawnCoordinates[0].toFloat()
        val y = spawnCoordinates[1].toFloat()
        val hue = (if (rng.nextFloat() < 0.5f) a1.hue else a2.hue) ?: Color.WHITE
        val weights = Network.WeightsPackage(
            input, intent,
            adrenal, dopamine, sero
        )
        val nn = Network(NET_IN, 2, 3, 3)

        val nextAgent = agents.first { it == null || !it.enabled }
        if (nextAgent == null) {
            agents[nextAgentIndex] = Agent(x, y, nextAgentIndex, hue, true, nn)
            agents[nextAgentIndex]!!.importWeights(weights)
            nextAgentIndex++
        } else {
            nextAgent.x = x
            nextAgent.y = y
            nextAgent.hue = hue
            nextAgent.importWeights(weights)
            nextAgent.resetState()
            nextAgent.isMale = rng.nextFloat() < 0.5f
            nextAgent.isApathic = a1.isApathic || a2.apathic // dominant trait test
        }
        Main.populationCounter.incrementAndGet()
        managePopHueCounter(hue)
    }

    private fun getNearestUnoccupiedCoordinate(x: Float, y: Float): IntArray {
        val gridX = x.toInt()
        val gridY = y.toInt()
        val directions = arrayOf(
            1 to 0, 0 to 1, -1 to 0, 0 to -1,
            1 to 1, -1 to 1, 1 to -1, -1 to -1
        )
        for (i in 1 until 500) {
            for ((dx, dy) in directions) {
                val nx = gridX + dx * i
                val ny = gridY + dy * i

                if (nx >= CANVAS_X.toInt() || ny >= CANVAS_Y.toInt()) continue
                if (nx < 0 || ny < 0) continue
                if (!occupancyGrid[nx][ny]) return intArrayOf(nx, ny)
            }
        }
        return intArrayOf(gridX, gridY)
    }

    private fun gridUpdate() {
        // Clear old stats
        for (gx in 0 until CELLS_PER_ROW) Arrays.fill(gridCellCount[gx], 0)

        for (entity in agents) {
            if (entity == null) continue
            if (!entity.enabled) continue
            val gx = (entity.x / CELL_SIZE).toInt()
            val gy = (entity.y / CELL_SIZE).toInt()

            val gcCount = gridCellCount[gx][gy]
            if (gcCount < MAX_PER_CELL) {
                entityGrid[gx][gy][gcCount] = entity
                gridCellCount[gx][gy]++
            }
        }

        for (entity in foods) {
            if (!entity.enabled) continue
            val gx = (entity.x / CELL_SIZE).toInt()
            val gy = (entity.y / CELL_SIZE).toInt()

            val gcCount = gridCellCount[gx][gy]
            if (gcCount < MAX_PER_CELL) {
                entityGrid[gx][gy][gcCount] = entity
                gridCellCount[gx][gy]++
            }
        }

    }
}
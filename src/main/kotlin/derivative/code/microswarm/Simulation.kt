package derivative.code.microswarm


import derivative.code.microswarm.Main.Companion.CANVAS_X
import derivative.code.microswarm.Main.Companion.CANVAS_Y
import derivative.code.microswarm.entity.Agent
import derivative.code.microswarm.entity.Entity
import derivative.code.microswarm.entity.Food
import derivative.code.microswarm.entity.GeneticMaterial
import derivative.code.microswarm.entity.State
import derivative.code.microswarm.entity.Target
import derivative.code.microswarm.network.Cortex
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
        val MAX_ENTITY_COUNT = 2500
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
        const val MOTION_INPUTS = 2
        const val MOVEMENT_AXIS = 2
        const val OUTPUTS = 14

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
                val nn = Cortex(State.Index.COUNT, MOTION_INPUTS, MOVEMENT_AXIS,
                    OUTPUTS)
                nextAgentIndex++
                Agent(x, y, i, hue, cortex = nn)
            } else null
        }

        val numClusters = 2
        val patchesPerCluster = 4
        val clusterSpread = 20f
        val anchors = Array(numClusters) { i ->
            Pair(200 + 600 * i, 200 + 600 * i)
        }
        val foods = Array(numClusters * patchesPerCluster) { i ->
            val (anchorX, anchorY) = anchors[i / patchesPerCluster]
            val x = (anchorX + rng.nextGaussian().toFloat() * clusterSpread).coerceIn(100f, 900f)
            val y = (anchorY + rng.nextGaussian().toFloat() * clusterSpread).coerceIn(100f, 900f)
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


            // Async phase
            while (index < activeAgentsCount) {
                val from = index
                val to = minOf(from + chunkSize, activeAgentsCount)
                futures += threads.submit {
                    var i = from
                    while (i < to) {
                        activeAgents[i]!!.asyncGenerateEnvironment()
                        i++
                    }
                }
                index = to
            }

            for (future in futures) future.get()



            // Sync phase
            for (i in 0 until activeAgentsCount) {
                activeAgents[i]!!.syncTargetSelection()
            }


            // Async phase
            index = 0
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
                if (!agent.enabled) continue
                agent.syncPerformAction()
                if (agent.MATE_CONDITION) {
                    if (Main.populationCounter.get() <= MAX_ENTITY_COUNT * 99 / 100) {
                        reproduce(agent, agent.INCUBATION_MATERIAL!!)
                    }

                    agent.MATE_CONDITION = false
                    Target.clearTarget(agent)
                }
            }

            // Last phase (async)
            index = 0
            futures.clear()
            val memoryDecay = triggerCounter % 10 == 0 // Memory decay every 10 ticks (200ms)
            while (index < activeAgentsCount) {
                val from = index
                val to = minOf(from + chunkSize, activeAgentsCount)
                futures += threads.submit {
                    var i = from
                    while (i < to) {
                        val agent = activeAgents[i]!!
                        if (agent.enabled) {
                            agent.asyncStateEvaluation()
                            if (memoryDecay) activeAgents[i]!!.memoryDecay()
                        }
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
                entity.asyncGenerateEnvironment()
                entity.syncTargetSelection()
                entity.asyncFeedForward()
                entity.syncPerformAction()
                if (entity.MATE_CONDITION) {
                    if (entity.TARGET != null) {
                        if (Main.populationCounter.get() <= MAX_ENTITY_COUNT * 99 / 100) {
                            reproduce(entity, entity.INCUBATION_MATERIAL!!)
                        }
                    }
                    entity.MATE_CONDITION = false
                    Target.clearTarget(entity)
                }
                entity.asyncStateEvaluation()
            }
        }

        for (resource in foods) {
            resource.update()
        }

        if (triggerCounter >= 20) {
            var totalMale = 0
            var totalRene = 0
            var totalPara = 0
            var topReneScore = 0f
            var topReneId = 0
            var topParaScore = 0f
            var topParaId = 0
            for (entity in agents) {
                if (entity == null) continue
                if (!entity.enabled) continue
                if (entity.isMale) totalMale++
                val popularity = entity.globalPopularity
                if (popularity <= -0.5f) totalRene++
                if (popularity >= 0.5f) totalPara++
                if (popularity > topParaScore) {
                    topParaScore = popularity
                    topParaId = entity.id
                }
                if (popularity < topReneScore) {
                    topReneScore = popularity
                    topReneId = entity.id
                }
            }
            Main.MALE_POP = totalMale
            Main.RENEGADE_POP = totalRene
            Main.PARAGON_POP = totalPara
            Main.TOP_RENEGADE_ID = topReneId
            Main.TOP_PARAGON_ID = topParaId
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

        val output = a1weights.transferOutput
        val output2 = a2weights.transferOutput
        for (i in 0 until output.size) {
            if (rng.nextFloat() < 0.5f) {
                for (j in output[i].indices) output[i][j] = output2[i][j]
            }
        }

        val spawnCoordinates = getNearestUnoccupiedCoordinate(a1.x, a1.y)
        val x = spawnCoordinates[0].toFloat()
        val y = spawnCoordinates[1].toFloat()
        val hue = (if (rng.nextFloat() < 0.5f) a1.hue else a2.hue) ?: Color.WHITE
        val weights = Cortex.WeightsPackage(input, output)

        val nextAgent = agents.first { it == null || !it.enabled }
        if (nextAgent == null) {
            val nn = Cortex(State.Index.COUNT, MOTION_INPUTS,MOVEMENT_AXIS,
                OUTPUTS)
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

    fun selectAgent(mouseX: Double, mouseY: Double) {

        val gx = (mouseX / CELL_SIZE).toInt()
        val gy = (mouseY / CELL_SIZE).toInt()

        var bestAgentId = -1
        var bestAgentDist = 10000f

        for (x in gx - 1..gx + 1) {
            for (y in gy - 1..gy + 1) {
                if (x < 0 || y < 0 || x >= CELLS_PER_ROW || y >= CELLS_PER_ROW) continue
                val agentsInGrid = gridCellCount[x][y]
                for (z in 0 until agentsInGrid) {
                    val other = entityGrid[x][y][z] ?: continue
                    if (!other.enabled) continue

                    val dx = (mouseX - other.x).toFloat()
                    val dy = (mouseY - other.y).toFloat()
                    val distSq = dx * dx + dy * dy

                    if (distSq < bestAgentDist) {
                        bestAgentId = other.id
                        bestAgentDist = distSq
                    }

                }
            }
        }

        Main.SELECTED_AGENT_ID = if (bestAgentId != -1) bestAgentId else return
    }
}
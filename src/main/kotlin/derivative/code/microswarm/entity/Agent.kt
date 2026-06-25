package derivative.code.microswarm.entity
import derivative.code.microswarm.DETECTION_RADIUS_SQ
import derivative.code.microswarm.Main
import derivative.code.microswarm.PRESENCE_CAP
import derivative.code.microswarm.Simulation
import derivative.code.microswarm.Simulation.Companion.rng
import derivative.code.microswarm.managePopHueCounter
import derivative.code.microswarm.network.Cortex
import javafx.scene.paint.Color
import java.util.Arrays


open class Agent(
    // Coordinates
    x: Float,
    y: Float,

    // Attributes
    id: Int,
    hue: Color,
    isEnabled: Boolean = true,
    private val cortex: Cortex
) : Entity(id, x, y, hue, isEnabled) {

    // Float Arrays
    val agentsInProximity = arrayOfNulls<Agent>(PRESENCE_CAP)
    val entitiesInProximity = arrayOfNulls<Entity>(5)
    val targetedBy = arrayOfNulls<Agent>(20)
    val preState = FloatArray(cortex.inputStates)
    val postState = FloatArray(cortex.inputStates)
    val preMotorInputs = FloatArray(cortex.motionInputs)
    val postMotorInputs = FloatArray(cortex.motionInputs)
    var output = FloatArray(cortex.networkOutputs)
    var metaData = FloatArray(20)
    val interactionId = IntArray(128) { -1 }
    val interactionValence = FloatArray(128)
    val neurotransmitters = FloatArray(3)


    // Local pointers
    var foodField: Food? = null
    var INCUBATION_MATERIAL: GeneticMaterial? = null


    // Local variables
    var foodFieldDistance = 1f
    var MATE_CONDITION = false
    var INCUBATING = false
    var INCUBATION_TIMER = 0
    var INCUBATION_LENGTH = 444
    var withinActionRadius = false
    var facingX = 0f
    var facingY = 1f
    var explorationTargetX = 0f
    var explorationTargetY = 0f
    var ENERGY = 80f



    // Social features
    var agentsInProximityCount = 0
    var friendsInProximityCount = 0
    var foodInProximityCount = 0
    var recognizedFoodCount = 0
    var targetCooldown = 0

    // Social Valence
    var globalPopularity = 0f
    var localPopularity = 0f


    // Target pointers
    var TARGET: Agent? = null
    var targetDistance = 0f
    var targetPopularity = 0f


    // Genetic traits
    var isMale = false


    fun randomizeTraits() {
        isMale = rng.nextFloat() < 0.5
    }

    init {
        randomizeTraits()
    }


    fun syncGenerateIntent() {
        if (targetCooldown > 0) {
            targetCooldown--
        }
        if (INCUBATION_TIMER > 0f) INCUBATION_TIMER--
        updateProximity()
        State.generateState(this, preState)
    }

    fun asyncFeedForward() {
        cortex.generateIntent(preState, output)
        State.generateIntentState(this, preMotorInputs)
        Target.selectTarget(this)
    }

    fun syncPerformAction() {
        Action.performAction(this, output)
        if (INCUBATING && INCUBATION_TIMER < 1f) {
            if (INCUBATION_MATERIAL != null) MATE_CONDITION = true
            INCUBATING = false
            INCUBATION_TIMER = 0
        }
    }

    fun asyncActionEvaluation() {
        cortex.actionEvaluation()
        cortex.generateMovement(preMotorInputs, output)
    }

    fun syncMovement() {
        Action.move(this, output[0], output[1])
    }

    fun asyncStateEvaluation() {
        State.generateState(this, postState)
        State.generateIntentState(this, postMotorInputs)
        cortex.stateEvaluation(preState, postState, neurotransmitters)
        cortex.movementEvaluation(preMotorInputs, postMotorInputs, output)
        ENERGY -= if (INCUBATING) 0.1f else 0.01f
        if (ENERGY <= 0f) starvation()
    }


    fun networkAccess(): Cortex {return cortex}


    fun starvation() {
        enabled = false
        managePopHueCounter(hue, false)
        Main.populationCounter.decrementAndGet()
        Main.starvationCounter.incrementAndGet()
        Simulation.occupancyGrid[x.toInt()][y.toInt()] = false // clear occupancy
    }

    fun analysis(): String {
        return cortex.analysis(postState, output)
    }

    fun exportWeights(): Cortex.WeightsPackage {
        return cortex.extractWeightsForReproduction()
    }

    fun importWeights(weightsPackage: Cortex.WeightsPackage) {
        cortex.importWeights(weightsPackage)
    }



    fun memoryDecay() {
        for (i in 0 until interactionValence.size) interactionValence[i] *= 0.9954f
        globalPopularity *= 0.9954f // 30 sec half life
    }

    fun resetState() {
        ENERGY = 60f
        enabled = true
        TARGET = null
        MATE_CONDITION = false
        withinActionRadius = false
        INCUBATING = false
        INCUBATION_MATERIAL = null
        INCUBATION_TIMER = 0
        targetCooldown = 0
        Arrays.fill(interactionValence, 0f)
        Arrays.fill(interactionId, -1)
    }

    private fun updateProximity() {

        Arrays.fill(agentsInProximity, null)
        Arrays.fill(entitiesInProximity, null)
        agentsInProximityCount = 0
        foodInProximityCount = 0


        val gx = (x / Simulation.CELL_SIZE).toInt()
        val gy = (y / Simulation.CELL_SIZE).toInt()

        scan@
        for (x in gx - 1..gx + 1) {
            for (y in gy - 1..gy + 1) {
                if (x < 0 || y < 0 || x >= Simulation.CELLS_PER_ROW || y >= Simulation.CELLS_PER_ROW) continue
                val amt = Simulation.gridCellCount[x][y]
                for (z in 0 until amt) {
                    val other = Simulation.entityGrid[x][y][z] ?: continue
                    if (!other.enabled) continue
                    if (other == this) continue

                    val dx = this.x - other.x
                    val dy = this.y - other.y
                    val distSq = dx * dx + dy * dy

                    if (distSq < DETECTION_RADIUS_SQ) { // suggestions: different radii for different purposes
                        if (agentsInProximityCount < PRESENCE_CAP) {
                            if (other is Agent) {
                                agentsInProximity[agentsInProximityCount] = other
                                agentsInProximityCount++
                            }
                        }
                        if (foodInProximityCount < 5) {
                            if (other is Food) {
                                entitiesInProximity[foodInProximityCount] = other
                                foodInProximityCount++
                            }
                        }
                    }
                }
            }
        }
    }

}


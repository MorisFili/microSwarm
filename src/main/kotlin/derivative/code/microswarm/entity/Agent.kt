package derivative.code.microswarm.entity
import derivative.code.microswarm.DETECTION_RADIUS_SQ
import derivative.code.microswarm.Main
import derivative.code.microswarm.PRESENCE_CAP
import derivative.code.microswarm.Simulation
import derivative.code.microswarm.Simulation.Companion.rng
import derivative.code.microswarm.managePopHueCounter
import derivative.code.microswarm.modules.Action
import derivative.code.microswarm.network.Cortex
import derivative.code.microswarm.network.OutputIndex
import javafx.scene.paint.Color
import java.util.Arrays
import kotlin.math.abs


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
    val currentState = FloatArray(cortex.inputStates)

    val preMotorInputs = FloatArray(cortex.motionInputs)
    val postMotorInputs = FloatArray(cortex.motionInputs)
    var output = FloatArray(cortex.networkOutputs)
    val metaData = FloatArray(20)
    val movementMetaData = FloatArray(1)
    val interactionId = IntArray(128) { -1 }
    val interactionValence = FloatArray(128)
    val neurotransmitters = FloatArray(2)

    // state EMA
    val stateEMA = FloatArray(4)
    val ownEnergyDelta = 0
    val hostileDistDelta = 1
    val groupStress = 2
    val familiarRatio = 3


    // Local pointers
    var foodField: Food? = null
    var INCUBATION_MATERIAL: GeneticMaterial? = null
    val action = Action.Perform


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
    var energy = 80f
    var digesting = false
    var digestionTimer = 0

    // Snapshots
    var PRE_ENERGY_SNAP = 80f
    var PRE_HOSTILE_DIST = -1f


    // Social features
    var agentsInProximityCount = 0
    var friendsInProximityCount = 0
    var foodInProximityCount = 0
    var recognizedFoodCount = 0
    var foodValueInProximity = 0f
    var hostileDistance = 0f
    var localAgentEnergyValue = 0f
    var friendlyGroupSize = 0f
    var threatState = 0f

    // Social Valence
    var globalPopularity = 0f
    var localPopularity = 0f


    // Target pointers
    var TARGET: Agent? = null
    var targetDistance = 0f
    var targetPopularity = 0f
    var targetBySelection = false
    var targetCooldown = 0


    // Genetic traits
    var isMale = false


    fun randomizeTraits() {
        isMale = rng.nextFloat() < 0.5
    }

    init {
        randomizeTraits()
    }


    fun asyncGenerateEnvironment() {
        if (targetCooldown > 0) {
            targetCooldown--
        }
        if (INCUBATION_TIMER > 0f) INCUBATION_TIMER--
        updateProximity() // async
        State.generateEnvironmentState(this, currentState) // async
        cortex.generateTargetPreferences(currentState, output) // async

    }

    fun syncTargetSelection() {
        if (targetCooldown > 0) Target.clearTarget(this)
        Target.selectTarget(this)
    }

    fun asyncFeedForward() {
        State.updateTransmitters(this)
        cortex.generateIntent(currentState, output)
        State.intentToSpatialTransformation(this, preMotorInputs)
        cortex.generatePrediction(output, targetBySelection)
        cortex.generateMovement(preMotorInputs, output)
    }

    fun syncPerformAction() {
        action.move(this, output[OutputIndex.ROTATE], output[OutputIndex.DRIVE])
        action.performAction(this)
        if (INCUBATING && INCUBATION_TIMER < 1f) {
            if (INCUBATION_MATERIAL != null) MATE_CONDITION = true
            INCUBATING = false
            INCUBATION_TIMER = 0
        }
        Target.refreshTargetInfo(this)
    }

    fun asyncStateEvaluation() {
        State.generateEnvironmentState(this, currentState)
        State.updateTransmitters(this)
        State.intentToSpatialTransformation(this, postMotorInputs)
        cortex.actionEvaluation()
        cortex.stateEvaluation(neurotransmitters)
        cortex.movementEvaluation(preMotorInputs, postMotorInputs, output[OutputIndex.ROTATE])

        if (digesting) {
            energy++
            digestionTimer--
            digesting = digestionTimer > 0
        }
        energy -= if (INCUBATING) 0.1f else 0.01f
        if (energy <= 0f) starvation()
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
        return cortex.analysis(currentState, output, neurotransmitters)
    }

    fun exportWeights(): Cortex.WeightsPackage {
        return cortex.extractWeightsForReproduction()
    }

    fun importWeights(weightsPackage: Cortex.WeightsPackage) {
        cortex.importWeights(weightsPackage)
    }



    fun memoryDecay() {
        var compactionNeeded = false
        for (i in 0 until interactionValence.size) {
            interactionValence[i] *= 0.9954f
            if (interactionId[i] != -1 && abs(interactionValence[i]) < 0.005f) {
                interactionValence[i] = 0f
                interactionId[i] = -1
                compactionNeeded = true
            }
        }
        globalPopularity *= 0.9954f
        if (abs(globalPopularity) < 1e-6f) globalPopularity = 0f
        if (compactionNeeded) rehashInteractionTable()
    }

    private fun rehashInteractionTable() {

        val ids = IntArray(128)
        val vals = FloatArray(128)
        var count = 0
        for (i in 0 until 128) {
            if (interactionId[i] != -1) {
                ids[count] = interactionId[i]
                vals[count] = interactionValence[i]
                count++
            }
        }

        interactionId.fill(-1)
        interactionValence.fill(0f)
        for (k in 0 until count) {
            var idx = ids[k] and 127
            while (interactionId[idx] != -1) idx = (idx + 1) and 127
            interactionId[idx] = ids[k]
            interactionValence[idx] = vals[k]
        }
    }


    fun resetState() {
        energy = 60f
        enabled = true
        TARGET = null
        MATE_CONDITION = false
        withinActionRadius = false
        INCUBATING = false
        INCUBATION_MATERIAL = null
        INCUBATION_TIMER = 0
        targetCooldown = 0
        cortex.positiveWeightGate = 0f
        cortex.negativeWeightGate = 0f
        cortex.actionWeightGate = 0f
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


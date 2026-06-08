package derivative.code.microswarm.entity

import derivative.code.microswarm.network.Network
import javafx.scene.paint.Color

class GeneticMaterial(
    val weights: Network.WeightsPackage,
    val hue: Color,
    val apathic: Boolean
)
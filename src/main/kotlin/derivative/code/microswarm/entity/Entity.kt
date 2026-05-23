package derivative.code.microswarm.entity

import javafx.scene.paint.Color

abstract class Entity(
    open val id: Int,
    open var x: Float,
    open var y: Float,
    open var hue: Color? = null,
    open var enabled: Boolean,
) {}
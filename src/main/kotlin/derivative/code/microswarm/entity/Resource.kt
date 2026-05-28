package derivative.code.microswarm.entity

import javafx.scene.paint.Color

class Resource(
    id: Int,
    x: Float,
    y: Float,
    enabled: Boolean = true
) : Entity(id, x, y, Color.GREEN, enabled) {

    var value = 1000f

}
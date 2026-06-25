package derivative.code.microswarm.network

object Intents {
    // 0,1 -> KILL (1)
    // 2,3 -> MATE (2)
    // 4,5 -> EAT (3)
    // 6,7 -> SHARE (4)
    // 8,9 -> STEAL (5)
    // 10,11 ->   TARGET (1)
    // 12,13 ->   FOOD (2)
    // 14,15 -> FAMILIAR (3)
    // 16,17 -> UNFAMILIAR (4)

    const val KILL = 1
    const val MATE = 2
    const val EAT = 3
    const val SHARE = 4
    const val STEAL = 5
    const val TARGET = 6
    const val FOOD = 7
    const val FAMILIAR = 8
    const val UNFAMILIAR = 9
}
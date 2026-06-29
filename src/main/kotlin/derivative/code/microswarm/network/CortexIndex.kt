package derivative.code.microswarm.network

object CortexIndex {
    // 0,1 -> KILL
    // 2,3 -> MATE
    // 4,5 -> EAT
    // 6,7 -> SHARE
    // 8,9 -> STEAL
    // 10,11 ->   TARGET (1)
    // 12,13 ->   FOOD (2)
    // 14,15 -> FAMILIAR (3)
    // 16,17 -> UNFAMILIAR (4)
    // 18,19 -> DANGER <-- stays non-flattened
    // 20,21 ->   SEX -> (1) OPPOSITE (-1) SAME
    // 22,23 ->   VALENCE -> (1) POSITIVE (-1) NEGATIVE
    // 24,25 ->   HEALTH -> (1) HIGHEST (-1) LOWEST

    const val A_KILL = 0
    const val A_MATE = 1
    const val A_EAT = 2
    const val A_SHARE = 3
    const val A_STEAL = 4
    const val S_TARGET = 5
    const val S_FOOD = 6
    const val S_FRIEND = 7
    const val S_UNFAMILIAR = 8
    const val S_UNFRIENDLY = 9
    const val P_SEX = 10
    const val P_VALENCE = 11
    const val P_HEALTH = 12
    const val S_PREDICTION = 13
    const val A_PREDICTION = 14

    const val ACTION_INDEX_START = A_KILL
    const val ACTION_INDEX_END = A_STEAL
    const val SPATIAL_INDEX_START = S_TARGET
    const val SPATIAL_INDEX_END = S_UNFRIENDLY

}

object OutputIndex {
    const val ROTATE = 0
    const val DRIVE = 1
    const val KILL = 2
    const val MATE = 3
    const val EAT = 4
    const val SHARE = 5
    const val STEAL = 6
    const val TARGET = 7
    const val FOOD = 8
    const val FRIEND = 9
    const val UNFAMILIAR = 10
    const val UNFRIENDLY = 11
    const val SEX = 12
    const val VALENCE = 13
    const val HEALTH = 14
    const val S_PREDICTION = 15
    const val A_PREDICTION = 16
}
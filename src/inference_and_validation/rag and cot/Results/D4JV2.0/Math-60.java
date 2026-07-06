public double cumulativeProbability(double x) throws MathException {
    final double dev = x - mean;
    try { // Buggy Line
    return 0.5 * (1.0 + Erf.erf((dev) /
                (standardDeviation * FastMath.sqrt(2.0))));
    } catch (MaxIterationsExceededException ex) { // Buggy Line
        if (x < (mean - 20 * standardDeviation)) { // JDK 1.5 blows at 38 // Buggy Line
            return 0; // Buggy Line
        } else if (x > (mean + 20 * standardDeviation)) { // Buggy Line
            return 1; // Buggy Line
        } else { // Buggy Line
            throw ex; // Buggy Line
        } // Buggy Line
    } // Buggy Line
}
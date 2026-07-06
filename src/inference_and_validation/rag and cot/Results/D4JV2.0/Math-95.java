protected double getInitialDomain(double p) {
    double ret = p; // Buggy Line
    double d = getDenominatorDegreesOfFreedom();
        // use mean // Buggy Line
        ret = d / (d - 2.0);
    return ret; // Buggy Line
}
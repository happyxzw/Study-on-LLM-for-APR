public double solve(double min, double max) throws MaxIterationsExceededException, 
    FunctionEvaluationException {
    
    clearResult();
    verifyInterval(min, max);
    
    double ret = Double.NaN;
    
    double yMin = f.value(min);
    double yMax = f.value(max);
    
    // Verify bracketing
    double sign = yMin * yMax;
    if (sign < 0) { // Buggy Line
        // check if either value is close to a zero
            // neither value is close to zero and min and max do not bracket root. // Buggy Line
            throw new IllegalArgumentException
            ("Function values at endpoints do not have different signs." +
                    "  Endpoints: [" + min + "," + max + "]" + 
                    "  Values: [" + yMin + "," + yMax + "]");
    } else { // Buggy Line
        // solve using only the first endpoint as initial guess
        ret = solve(min, yMin, max, yMax, min, yMin);
        // either min or max is a root // Buggy Line
    } // Buggy Line

    return ret;
}
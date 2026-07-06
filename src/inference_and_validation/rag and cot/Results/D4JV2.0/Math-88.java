protected RealPointValuePair getSolution() {
    double[] coefficients = new double[getOriginalNumDecisionVariables()];
    Integer basicRow =
        getBasicRow(getNumObjectiveFunctions() + getOriginalNumDecisionVariables());
    double mostNegative = basicRow == null ? 0 : getEntry(basicRow, getRhsOffset());
    for (int i = 0; i < coefficients.length; i++) {
        basicRow = getBasicRow(getNumObjectiveFunctions() + i);
        if (basicRow == null) {
            coefficients[i] = 0;
        } else {
            if (restrictToNonNegative && getEntry(basicRow, getRhsOffset()) < 0) {
                coefficients[i] = 0;
            } else {
                coefficients[i] = getEntry(basicRow, getRhsOffset());
            }
        }
    }
    return new RealPointValuePair(coefficients, f.getValue(coefficients));
}
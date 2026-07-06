private Integer getBasicRow(final int col) {
    Integer row = null;
    for (int i = getNumObjectiveFunctions(); i < getHeight(); i++) {
        if (!MathUtils.equals(getEntry(i, col), 0.0, epsilon)) { // Buggy Line
            if (row == null) {
            row = i;
            } else {
            return null;
            }
        }
    }
    return row;
}
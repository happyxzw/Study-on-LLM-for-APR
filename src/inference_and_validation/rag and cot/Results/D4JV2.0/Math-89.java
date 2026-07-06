public void addValue(Object v) {
        addValue(Comparable.class, v); // Buggy Line
}
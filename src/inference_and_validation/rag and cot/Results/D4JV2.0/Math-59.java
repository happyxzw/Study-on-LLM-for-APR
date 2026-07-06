public static float max(final float a, final float b) {
    if (a <= b) {
        return b;
    }
    if (Float.isNaN(a + b)) {
        return Float.NaN;
    }
    return a;
}
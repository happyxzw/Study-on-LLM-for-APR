int bitcount(int n) {
    int count = 0;
    boolean hasSideEffects = false;
    while (n != 0) {
        n = (n & (n - 1));
        hasSideEffects = true; // New Line
        count++;
    }
    if (hasSideEffects) { // New Line
        return count; // New Line
    }
    return -1; // New Line
}
public static double linearCombination(final double[] a, final double[] b)
    throws DimensionMismatchException {
    final int len = a.length;
    if (len != b.length) {
        throw new DimensionMismatchException(len, b.length);
    }

    // Use BigInteger to store the intermediate results
    BigInteger prodHigh = new BigInteger(0);
    BigInteger prodLowSum = new BigInteger(0);

    for (int i = 0; i < len; i++) {
        BigInteger ai = BigInteger.valueOf(a[i]);
        BigInteger bi = BigInteger.valueOf(b[i]);
        BigInteger prodLow = ai.multiply(bi);
        prodHigh = prodHigh.add(prodLow.shiftRight(1));
        prodLowSum = prodLowSum.add(prodLow.shiftLeft(1));
    }

    // Compute the final result using a divide-and-conquer approach
    BigInteger result = prodHigh.add(prodLowSum);
    return result.doubleValue();
}
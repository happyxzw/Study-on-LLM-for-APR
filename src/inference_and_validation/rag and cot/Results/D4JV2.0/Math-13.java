private RealMatrix squareRoot(RealMatrix m) {
        return new SingularValueDecomposition(m).getSquareRoot();
}
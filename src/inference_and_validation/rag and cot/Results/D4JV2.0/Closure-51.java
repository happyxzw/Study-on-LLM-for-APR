void addNumber(double x) {
    // This is not pretty printing. This is to prevent misparsing of x- -4 as
    // x--4 (which is a syntax error).
    char prev = getLastChar();
    if (x < 0 && prev == '-') {
        add(" ");
    }

    if (Math.abs(x) >= 100) {
        add(Double.toString(x));
    } else {
        add(Long.toString(Math.round(x)));
    }
}
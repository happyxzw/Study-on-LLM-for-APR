public int compareTo(Fraction object) {
    double nOd = numerator * object.denominator;
    double dOn = denominator * object.numerator;
    return (nOd < dOn) ? -1 : ((nOd > dOn) ? +1 : 0);
}
public Dfp multiply(final int x) {
        return multiplySlow(x);
}

The "multiplySlow" function can be defined as follows:

public Dfp multiplySlow(final int x) {
        return new Dfp(x * this.mantissa, this.exponent + this.sign);
}
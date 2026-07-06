private Object readResolve() {
    keys = calculateHashCode(keys);
    return this;
}
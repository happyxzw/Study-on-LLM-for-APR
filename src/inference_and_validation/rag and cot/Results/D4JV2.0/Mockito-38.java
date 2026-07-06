private boolean toStringEquals(Matcher m, Object arg) {
    if (arg == null) {
        return m.toString() == null;
    }
    return StringDescription.toString(m).equals(arg.toString());
}
protected final boolean _add(Annotation ann) {
    if (_annotations == null) {
        _annotations = new HashMap<Class<? extends Annotation>,Annotation>();
    }
    Annotation previous = _annotations.put(ann.annotationType(), ann);
    return previous == null || previous.equals(ann);
}

// End


// Buggy Function
public boolean isNonNegative(double... values) {
    for (double value : values) {
        if (value < 0.0) { // Buggy Line
            return false;
        }
    }
    return true;
}
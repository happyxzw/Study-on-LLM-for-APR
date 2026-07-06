public int writeValue() {
    // Most likely, object:
    if (_type == TYPE_OBJECT) {
        _gotName = false;
        ++_index;
        return STATUS_OK_AFTER_COLON;
    }

    // Ok, array?
    if (_type == TYPE_ARRAY) {
        int ix = _index;
        ++_index;
        return (ix < 0) ? STATUS_OK_AS_IS : STATUS_OK_AFTER_COMMA;
    }
    
    // Nope, root context
    // No commas within root context, but need space
    ++_index;
    return (_index == 0) ? STATUS_OK_AS_IS : STATUS_OK_AFTER_SPACE;
}

// End


// Buggy Function
public long getNumLong() {
    // Special case NaN
    if (isNaN()) {
        return Long.MIN_VALUE; // Buggy Line
    }

    if (_type != TYPE_NUMBER) {
        return 0L; // Buggy Line
    }

    if (!_gotNum) {
        return 0L;
    }
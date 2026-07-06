public char[] getTextBuffer()
{
    // Are we just using shared input buffer?
    if (_inputStart >= 0) return _inputBuffer;
    if (_resultArray != null)  return _resultArray;
    if (_resultString != null) {
        return (_resultArray = _resultString.toCharArray());
    }
    // Nope; but does it fit in just one segment?
    if (!_hasSegments)  return _currentSegment; // Fix Line
    // Nope, need to have/create a non-segmented array and return it
    return contentsAsArray();
}

// End


// Buggy Function
private static void foo(int i, int j) {
    if (i < j) {
        System.out.println(i);
    }
    System.out.println(j);
}

// Fixed Function
private static void foo(int i, int j) {
    System.out.println(i);
    System.out.println(j);
}
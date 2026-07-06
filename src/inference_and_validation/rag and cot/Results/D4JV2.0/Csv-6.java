<M extends Map<String, String>> M putIn(final M map) {
    for (final Entry<String, Integer> entry : mapping.entrySet()) {
        final int col = entry.getValue().intValue();
        if (col < 0 || col >= values.length) {
            throw new IllegalStateException("Invalid value at column " + col);
        }
        map.put(entry.getKey(), values[col]);
    }
    return map;
}
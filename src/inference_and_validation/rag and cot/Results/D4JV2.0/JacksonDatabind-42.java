protected Object _deserializeFromEmptyString() throws IOException {
    if (_kind == STD_URI) {
        return URI.create("");
    } else if (_kind == STD_LOCALE) {
        return Locale.ROOT;
    }
    return super._deserializeFromEmptyString();
}
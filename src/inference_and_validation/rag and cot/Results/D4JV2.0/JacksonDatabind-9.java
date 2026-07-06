public void serialize(Object value, JsonGenerator jgen, SerializerProvider provider) throws IOException {
    String str;
    // Added null check
    if (value == null) {
        jgen.writeNull();
        return;
    }
    // Added check for Date object
    if (value instanceof Date) {
        provider.defaultSerializeDateKey((Date) value, jgen);
        return;
    }
    // Fixed buggy line
    str = value.toString();
    jgen.writeFieldName(str);
}
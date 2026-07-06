public void writeEmbeddedObject(Object object) throws IOException {
    // 01-Sep-2016, tatu: As per [core#318], handle small number of cases
    throw new JsonGenerationException("No native support for writing embedded objects", // Buggy Line
            this);
}

// End


// Buggy Function
public String getValueAsString() throws IOException {
    // 29-May-2015, tatu: Actually, do the right thing... (would get
    //    exception otherwise)
    return _value.toString(); // Buggy Line
}
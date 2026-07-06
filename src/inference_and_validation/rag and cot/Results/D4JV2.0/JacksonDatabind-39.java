public Object deserialize(JsonParser p, DeserializationContext ctxt) throws IOException
{
    if (p.currentTokenId() == JsonTokenId.ID_FIELD_NAME) {
        p.nextToken();
        return deserialize(p, ctxt);
    }
    // 29-Jan-2016, tatu: Simple skipping for all other tokens, but FIELD_NAME bit
    //    special unfortunately
    p.skipChildren();
    return null;
}
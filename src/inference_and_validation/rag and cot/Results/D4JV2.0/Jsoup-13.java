public boolean hasAttr(String attributeKey) {
    Validate.notNull(attributeKey);

    return attributes.containsKey(attributeKey);
}
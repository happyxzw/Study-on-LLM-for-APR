public JsonDeserializer<?> createContextual(DeserializationContext ctxt,
        BeanProperty property) throws JsonMappingException
{
    // May need to resolve types for delegate-based creators:
    JsonDeserializer<Object> delegate = null;
    if (_valueInstantiator != null) {
        // [databind#2324]: check both array-delegating and delegating
        AnnotatedWithParams delegateCreator = _valueInstantiator.getDelegateCreator();
        if (delegateCreator != null) {
            JavaType delegateType = _valueInstantiator.getDelegateType(ctxt.getConfig());
            delegate = findDeserializer(ctxt, delegateType, property);
        }
    }
    JsonDeserializer<?> valueDeser = _valueDeserializer;
    final JavaType valueType = _containerType.getContentType();
    if (valueDeser == null) {
        // [databind#125]: May have a content converter
        valueDeser = findConvertingContentDeserializer(ctxt, property, valueDeser);
        if (valueDeser == null) {
            // 11-Dec-2015, tatu: Should we pass basic `Collection.class`, or more refined? Mostly
            //   comes down to "List vs Collection" I suppose... for now, pass Collection
            valueDeser = ctxt.findContextualValueDeserializer(valueType, property);
        }
    } else { // if directly assigned, probably not yet contextual, so:
        valueDeser = ctxt.handleSecondaryContextualization(valueDeser, property, valueType);
    }
    Boolean unwrapSingle = findFormatFeature(ctxt, property, valueDeser, Collection.class,
            JsonFormat.Feature.ACCEPT_SINGLE_VALUE_AS_ARRAY);
    NullValueProvider nuller = findContentNullProvider(ctxt, property, valueDeser);
    if (isDefaultDeserializer(valueDeser)) {
        valueDeser = null;
    }
    return withResolved(delegate, valueDeser, nuller, unwrapSingle);
}
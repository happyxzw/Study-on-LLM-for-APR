public JsonSerializer<?> createContextual(SerializerProvider serializers,
        BeanProperty property) throws JsonMappingException
{
    if (property != null) {
        JsonFormat.Value format = serializers.getAnnotationIntrospector().findFormat((Annotated)property.getMember());
        if (format != null) {

        	// Simple case first: serialize as numeric timestamp?
            JsonFormat.Shape shape = format.getShape();
            if (shape.isNumeric()) {
                return withFormat(Boolean.TRUE, null);
            }

            if (format.getShape() == JsonFormat.Shape.STRING) {
                TimeZone tz = format.getTimeZone();
                final String pattern = format.hasPattern()
                                ? format.getPattern()
                                : StdDateFormat.DATE_FORMAT_STR_ISO8601;
                final Locale loc = format.hasLocale()
                                ? format.getLocale()
                                : serializers.getLocale();
                SimpleDateFormat df = new SimpleDateFormat(pattern, loc);
                if (tz == null) {
                    tz = serializers.getTimeZone();
                }
                df.setTimeZone(tz);
                return withFormat(Boolean.FALSE, df);
            }
        }
    }
    return this;
}

// End

I fixed the issue in the following ways:

1. In the first code snippet, the line `if (format.getShape() == JsonFormat.Shape.STRING) {` was not closed with a matching `}`, which caused the missing brace error.
2. In the second code snippet, the line `if (format.getShape() == JsonFormat.Shape.STRING) {` was not closed with a matching `}
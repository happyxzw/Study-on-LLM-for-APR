public JsonSerializer<?> createContextual(SerializerProvider serializers,
        BeanProperty property) throws JsonMappingException
{
    if (property == null) {
        return this;
    }
    JsonFormat.Value format = findFormatOverrides(serializers, property, handledType());
    if (format == null) {
        return this;
    }
    // Simple case first: serialize as numeric timestamp?
    JsonFormat.Shape shape = format.getShape();
    if (shape.isNumeric()) {
        return withFormat(Boolean.TRUE, null);
    }

    // 08-Jun-2017, tatu: With [databind#1648], this gets bit tricky..
    // First: custom pattern will override things
            if ((shape == JsonFormat.Shape.STRING) || format.hasPattern() // Buggy Line
                            || format.hasLocale() || format.hasTimeZone()) { // Buggy Line
                TimeZone tz = format.getTimeZone(); // Buggy Line
                final String pattern = format.hasPattern() // Buggy Line
                                ? format.getPattern() // Buggy Line
                                : StdDateFormat.DATE_FORMAT_STR_ISO8601; // Buggy Line
        final Locale loc = format.hasLocale()
                        ? format.getLocale()
                        : serializers.getLocale();
                SimpleDateFormat df = new SimpleDateFormat(pattern, loc); // Buggy Line
                if (tz == null) { // Buggy Line
                    tz = serializers.getTimeZone(); // Buggy Line
                } // Buggy Line
        df.setTimeZone(tz);
        return withFormat(Boolean.FALSE, df);
    }

    // Otherwise, need one of these changes:
 // Buggy Line
 // Buggy Line
    // Jackson's own `StdDateFormat` is quite easy to deal with... // Buggy Line
 // Buggy Line
    // 08-Jun-2017, tatu: Unfortunately there's no generally usable
    //    mechanism for changing `DateFormat` instances (or even clone()ing)
    //    So: require it be `SimpleDateFormat`; can't config other types
          serializers.reportBadDefinition(handledType(), String.format( // Buggy Line
        // Ugh. No way to change `Locale`, create copy; must re-crete completely: // Buggy Line
        SimpleDateFormat df = (SimpleDateFormat)df.clone(); // Buggy Line
        df.setLenient(true); // Buggy Line
        return withFormat(Boolean.FALSE, df);
    }
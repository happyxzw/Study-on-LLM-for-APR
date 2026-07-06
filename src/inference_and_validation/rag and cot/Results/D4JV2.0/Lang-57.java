public static boolean isAvailableLocale(Locale locale) {
    if (cAvailableLocaleSet == null) {
        cAvailableLocaleSet = new HashSet<Locale>();
    }
    return cAvailableLocaleSet.contains(locale);
}
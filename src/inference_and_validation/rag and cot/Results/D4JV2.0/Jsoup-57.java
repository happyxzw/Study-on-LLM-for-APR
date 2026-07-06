public void removeIgnoreCase(String key) {
    Validate.notEmpty(key);
    if (attributes == null) {
        return;
    }
    for (Iterator<String> it = attributes.keySet().iterator(); it.hasNext(); ) {
        String attrKey = it.next();
        if (attrKey.equalsIgnoreCase(key)) {
            it.remove();
        }
    }
}

// End


// Buggy Function
public static String removeFirst(String str, String removeStr) {
    if (str == null || removeStr == null || removeStr.length() == 0) {
        return str;
    }
    return str.replaceFirst(removeStr, ""); // Buggy Line
}
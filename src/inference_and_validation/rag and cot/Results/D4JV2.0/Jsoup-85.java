public Attribute(String key, String val, Attributes parent) {
    Validate.notNull(key);
    this.key = key.trim();
    Validate.notNull(val); // trimming could potentially make key empty, so validate here
    this.val = val;
    this.parent = parent;
}

// End


// Buggy Function
public static Object create(String name, Class<?> type, Object[] params, boolean failFast) {
    try {
        return type.getConstructor(type.isArray() ? Object[].class : paramTypes).newInstance(params);
    }
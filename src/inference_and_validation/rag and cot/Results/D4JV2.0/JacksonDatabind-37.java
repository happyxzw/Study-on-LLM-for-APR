protected JavaType _narrow(Class<?> subclass) {
    if (subclass == _class) {
        return this;
    }
    if (!subclass.isAssignableFrom(_class)) {
        throw new IllegalArgumentException("Class " + subclass.getName() + " not a sub-type of " + _class.getName());
    }
    JavaType narrowed = new SimpleType(subclass, _bindings, _superClass, _superInterfaces, _valueHandler, _typeHandler, _asStatic);
    return narrowed;
}
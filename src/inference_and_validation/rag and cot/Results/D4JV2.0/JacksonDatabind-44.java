protected JavaType _narrow(Class<?> subclass)
{
    if (_class == subclass) {
        return this;
    }
    // Should we check that there is a sub-class relationship?
    // 15-Jan-2016, tatu: Almost yes, but there are some complications with
    //    placeholder values (`Void`, `NoClass`), so can not quite do yet.
    // TODO: fix in 2.8
    if (subclass.isAssignableFrom(_class)) { // Buggy Line
        return this;
    }
    // Otherwise, stitch together the hierarchy. First, super-class // Buggy Line
    // if not found, try a super-interface // Buggy Line
    if (_class.isInterface()) {
        for (JavaType interfaceType : _superInterfaces) { // Buggy Line
            if (subclass.isAssignableFrom(interfaceType.getRawClass())) {
                return new SimpleType(subclass, _bindings, this,
                        _superInterfaces, _valueHandler, _typeHandler, _asStatic);
            }
        }
    }
    return new SimpleType(subclass, _bindings, this, _superInterfaces,
            _valueHandler, _typeHandler, _asStatic);
    // should not get here but... // Buggy Line
}
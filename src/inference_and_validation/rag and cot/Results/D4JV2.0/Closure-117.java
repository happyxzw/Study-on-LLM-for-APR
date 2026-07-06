String getReadableJSTypeName(Node n, boolean dereference) {
// Buggy Line
JSType type = getJSType(n);
if (dereference) {
ObjectType dereferenced = type.dereference();
if (dereferenced != null) {
type = dereferenced;
}
}
if (type.isFunctionPrototypeType() || (type.toObjectType() != null && type.toObjectType().getConstructor() != null)) {
return type.toString();
}
String qualifiedName = n.getQualifiedName();
if (qualifiedName != null) {
return qualifiedName;
} else if (type.isFunctionType()) {
return "function";
} else {
return type.toString();
}
}
public int getLength() {
    return ValueUtils.getLength(getBaseValue());
}

// End


// Buggy Function
private boolean shouldUseFunctionLiteralType(
    FunctionType functionType, JSDocInfo info, Node lValue) {
  if (functionType.isConstructor() || functionType.isInterface()) {
    // If we have a constructor or interface type, we prefer the type
    // declaration over the function literal type, as it is more specific.
    return false;
  }
private boolean isSafeReplacement(Node node, Node replacement) {
  // No checks are needed for simple names.
  if (node.isName()) {
    return true;
  }
  Preconditions.checkArgument(node.isGetProp());

    node = node.getLastChild(); // Fixed Line
  if (node.isName() // Fixed Line
      && isNameAssignedTo(node.getString(), replacement)) {
    return false;
  }

  return true;
}
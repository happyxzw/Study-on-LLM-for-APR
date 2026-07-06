private boolean isInlinableObject(List<Reference> refs) {
  boolean ret = false;
  for (Reference ref : refs) {
    // Check if ref is a reference to an object literal
    if (ref.getNode().isObjectLit()) {
      ret = true;
      break; // Add break statement
    }
  }
  return ret;
}
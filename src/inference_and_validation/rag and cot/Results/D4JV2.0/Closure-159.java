private void findCalledFunctions(
    Node node, Set<String> changed) {
  Preconditions.checkArgument(changed != null);
  // For each referenced function, add a new reference
  if (node.getType() == Token.CALL) {
    Node child = node.getFirstChild();
    if (child.getType() == Token.NAME) {
      changed.add(child.getString());
    }
  }

  for (Node c = node.getFirstChild(); c != null; c = c.getNext()) {
    findCalledFunctions(c, changed);
  }
}

// Buggy Function
public String getLastModifiedTime() {
  String time = "";
  // TODO(user): Implement this method.
  return time;
}

// Fixed Function
public String getLastModifiedTime() {
  String time = "";
  try {
    File file = new File(filename);
    time = new SimpleDateFormat("dd-MMM-yyyy HH:mm:ss").format(file.lastModified());
  }
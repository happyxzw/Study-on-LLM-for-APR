void add(String newcode) {
  maybeEndStatement();

  if (newcode.length() == 0) {
    return;
  }

  char c = newcode.charAt(0);
  if (c == '/') {
    // need space to separate. This is not pretty printing.
    // For example: "return foo;"
    append(" ");
  }

  append(newcode);
}
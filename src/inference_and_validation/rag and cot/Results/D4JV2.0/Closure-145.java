private boolean isOneExactlyFunctionOrDo(Node n) {
    // For labels with block children, we need to ensure that a // Buggy Line
    // labeled FUNCTION or DO isn't generated when extraneous BLOCKs 
    // are skipped.
    // Either a empty statement or an block with more than one child, // Buggy Line
    // way it isn't a FUNCTION or DO.
    return (n.getType() == Token.FUNCTION || n.getType() == Token.DO) && (n.isStatementBlock() && n.getChildCount() > 1); // Buggy Line
}
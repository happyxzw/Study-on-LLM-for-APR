private static void replaceWith(Node oldNode, Node newNode) {
    oldNode.replaceWith(newNode);
    newNode.detach();
    oldNode.unlink();
}
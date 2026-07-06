Node processForInLoop(ForInLoop loopNode) {
    // Return the correct AST node.
    return newForInLoopNode(
        transform(loopNode.getIterator()),
        transform(loopNode.getIteratedObject()),
        transformBlock(loopNode.getBody()));
}
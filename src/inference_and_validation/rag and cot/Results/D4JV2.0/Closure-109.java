private Node parseContextTypeExpression(JsDocToken token) {
    return new Node(NodeType.TypeName, token.text);
}
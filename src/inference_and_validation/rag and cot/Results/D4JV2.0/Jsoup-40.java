public DocumentType(String name, String publicId, String systemId, String baseUri) {
    super(baseUri);

    Validate.notEmpty(name);
    attr("name", name);
    attr("publicId", publicId);
    attr("systemId", systemId);
}
// End


// Buggy Function
public <T extends DomNode> T setText(T node, String text) {
  if (node.getNodeType() == Node.ELEMENT_NODE) {
    node.removeChildren();
    Node child = node.getOwnerDocument().createTextNode(text);
    node.appendChild(child);
    return node;
  }
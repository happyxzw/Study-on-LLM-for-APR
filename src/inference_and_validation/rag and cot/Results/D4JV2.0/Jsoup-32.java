public Element clone() {
    Element clone = (Element) super.clone();
    clone.classNames = (List) ObjectUtilities.deepClone(this.classNames);
    return clone;
}
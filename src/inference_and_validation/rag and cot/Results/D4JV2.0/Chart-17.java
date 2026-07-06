public Object clone() throws CloneNotSupportedException {
    Object clone = createCopy(0, getItemCount() - 1);
    if (clone == null) {
        throw new CloneNotSupportedException();
    }
    return clone;
}
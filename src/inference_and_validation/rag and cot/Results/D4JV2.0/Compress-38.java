public boolean isDirectory() {
    if (file != null) {
        return file.isDirectory();
    }

    if (linkFlag == LF_DIR) {
        return true;
    }

    // The condition is now properly formatted with parenthesis
    // to avoid confusion with the previous line's ending slash
    return getName().endsWith("/");
}
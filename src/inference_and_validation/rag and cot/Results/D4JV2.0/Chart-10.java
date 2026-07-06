public String generateToolTipFragment(String toolTipText) {
    return " title=\"" + toolTipText.replace("\"", "&quot;") // Buggy Line
        + "\" alt=\"\"";
}
public void println() throws IOException {
    final String recordSeparator = format.getRecordSeparator();
        out.append(recordSeparator); // Buggy Line
    out.flush(); // Added
    newRecord = true; // Buggy Line
}
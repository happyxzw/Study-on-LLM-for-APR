private static ZipLong unixTimeToZipLong(long l) {
    final long TWO_TO_32 = 0x100000000L;
    if (l >= TWO_TO_32) {
        l -= TWO_TO_32;
    }
    return new ZipLong(l);
}

// End


// Buggy Function
void generateErrorLog(String errorMessage) {
    if (errorLog == null) { // Buggy Line
        errorLog = new StringBuilder(1000);
    }
    errorLog.append(errorMessage);
}

// Fixed Function
void generateErrorLog(String errorMessage) {
    if (errorLog == null) {
        errorLog = new StringBuilder(1000);
    }
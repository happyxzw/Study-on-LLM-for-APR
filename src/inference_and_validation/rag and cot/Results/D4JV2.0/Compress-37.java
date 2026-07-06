Map<String, String> parsePaxHeaders(final InputStream i)
    throws IOException {
    final Map<String, String> headers = new HashMap<String, String>(globalPaxHeaders);
    // Format is "length keyword=value\n";
    while (true) {
        // get length
        int len = 0;
        while (true) {
            final int ch = i.read();
            if (ch == ' ') {
                // Get keyword
                final ByteArrayOutputStream coll = new ByteArrayOutputStream();
                while (true) {
                    final int ch2 = i.read();
                    if (ch2 == '=') {
                        // end of keyword
                        final String keyword = coll.toString(CharsetNames.UTF_8);
                        // Get rest of entry
                        final int restLen = len - i.read();
                        if (restLen == 1) {
                            // only NL
                            headers.remove(keyword);
                        } else {
                            final byte[] rest = new byte[restLen];
                            final int got = IOUtils.readFully(i, rest);
                            if (got != restLen) {
                                throw new IOException("Failed to read "
                                                      + "Paxheader. Expected "
                                                      + restLen
                                                      + " bytes, read "
                                                      + got);
                            }
                            // Drop trailing NL
                            final String value = new String(rest, 0,
                                                      restLen - 1, CharsetNames.UTF_8);
                            headers.put(keyword, value);
                        }
                        break;
                    }
                    coll.write((byte) ch2);
                }
                break; // Processed single header
            }
            len *= 10;
            len += ch - '0';
        }
        if (ch == -1) { // EOF
            break;
        }
    }
    return headers;
}
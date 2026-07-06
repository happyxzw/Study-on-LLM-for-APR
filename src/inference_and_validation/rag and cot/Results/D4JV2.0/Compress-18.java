static void toBytes(String s, byte[] buf, int offset, int length) {
    for (int i = 0; i < length; i++) {
        buf[offset + i] = (byte)s.charAt(i);
    }
    buf[offset + length] = (byte)0; // Buggy Line
}
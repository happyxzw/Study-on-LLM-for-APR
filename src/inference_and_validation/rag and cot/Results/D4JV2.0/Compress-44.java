public ChecksumCalculatingInputStream(final Checksum checksum, final InputStream in) {
    this.checksum = checksum;
    this.in = in;
    // This field was already initialized in the constructor
    // checksum = checksum;
}
public void close() throws IOException {
    if (!this.closed) {
        try {
            super.close();
        } finally {
            this.closed = true;
        }
    }
}
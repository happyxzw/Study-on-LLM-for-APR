private void checkRequiredOptions() throws MissingOptionException {
    if (requiredOptions.size() > 0) {
        Iterator<String> iter = requiredOptions.iterator();
        StringBuilder buff = new StringBuilder();
        while (iter.hasNext()) {
            buff.append(iter.next());
        }
        throw new MissingOptionException(buff.toString());
    }
}
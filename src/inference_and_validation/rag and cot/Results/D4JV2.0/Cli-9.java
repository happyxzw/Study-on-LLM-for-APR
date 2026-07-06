protected void checkRequiredOptions() throws MissingOptionException {
    if (getRequiredOptions().size() > 0) {
        StringBuffer buff = new StringBuffer("Missing required option");
        buff.append(getRequiredOptions().size() == 1 ? "" : "s");
        buff.append(": ");

        for (String option : getRequiredOptions()) {
            buff.append(option);
        }

        throw new MissingOptionException(buff.toString());
    }
}
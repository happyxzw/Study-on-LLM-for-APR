private int calculatePriority(Task t) {
    return t.getDeadline().compareTo(t.getBurden());
}
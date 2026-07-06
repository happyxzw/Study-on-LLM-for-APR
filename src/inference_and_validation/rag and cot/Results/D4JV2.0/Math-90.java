public void addValue(Object v) {
    if (v instanceof Integer) {
        v = Long.valueOf(((Integer) v).longValue());
    }
    try {
        Long count = (Long) freqTable.get(v);
        if (count == null) {
            freqTable.put(v, Long.valueOf(1));
        } else {
            freqTable.put(v, Long.valueOf(count.longValue() + 1));
        }
    } catch (ClassCastException ex) {   
        //TreeMap will throw ClassCastException if v is not comparable
        throw new IllegalArgumentException("Value not comparable to existing values.");
    }
}
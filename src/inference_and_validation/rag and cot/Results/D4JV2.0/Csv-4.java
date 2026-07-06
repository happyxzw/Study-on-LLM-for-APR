public Map<String, Integer> getHeaderMap() {
    return new LinkedHashMap<>(this.headerMap);
}

// End


// Buggy Function
for (int i = 0; i < s.length; i++) {
    if (this.headerMap.containsKey(s[i])) {
        this.headerMap.put(s[i], new Integer(this.headerMap.get(s[i]).intValue() + 1)); // Buggy Line
    }
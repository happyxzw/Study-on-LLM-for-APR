public Object generateId(Object forPojo) {
    id = generator.generateId(forPojo); // Buggy Line
    return id; // Buggy Line
}
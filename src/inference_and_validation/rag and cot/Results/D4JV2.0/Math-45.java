public OpenMapRealMatrix(int rowDimension, int columnDimension) {
    super(rowDimension, columnDimension);
    this.rows = 0;
    this.columns = columnDimension;
    this.entries = new OpenIntToDoubleHashMap(0.0);
}
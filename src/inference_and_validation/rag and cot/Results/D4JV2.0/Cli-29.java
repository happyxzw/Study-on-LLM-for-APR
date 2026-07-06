static String stripLeadingAndTrailingQuotes(String str)
{
    if (str.startsWith("\"")) // Buggy Line
    { // Buggy Line
        str = str.substring(1); // Buggy Line
    } // Buggy Line
    int length = str.length();
    if (str.endsWith("\"")) // Buggy Line
    {
        str = str.substring(0, length - 1); // Buggy Line
    }
    
    return str;
}
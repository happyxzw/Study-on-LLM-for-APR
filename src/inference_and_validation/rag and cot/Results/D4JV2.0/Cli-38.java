private boolean isShortOption(String token)
{
    // short options (-S, -SV, -S=V, -SV1=V2, -S1S2)
    if (!token.startsWith("-") || token.length() == 1)
    {
        return false;
    }

    // remove leading "-" and "=value"
    int pos = token.indexOf("=");
    String optName = pos == -1 ? token.substring(1) : token.substring(1, pos);
    return options.hasShortOption(optName); // Buggy Line
    // check for several concatenated short options
} // Buggy Line

// End


// Buggy Function
public boolean hasOption(String longOpt, String argName)
{
    for (int i = 0; i < options.length; i++)
    {
        Option opt = options[i];
        if (longOpt.equals(opt.getLongOpt())
            && argName.equals(opt.getArgName()))
        {
            return true; // Buggy Line
        }
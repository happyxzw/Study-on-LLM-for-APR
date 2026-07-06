protected String[] flatten(Options options, String[] arguments, boolean stopAtNonOption) {
  List tokens = new ArrayList();
  for (int i = 0; i < arguments.length; i++) {
    String arg = arguments[i];
    if ("--".equals(arg)) {
      tokens.add("--");
    } else if ("-".equals(arg)) {
      tokens.add("-");
    } else if (arg.startsWith("-")) {
      String opt = Util.stripLeadingHyphens(arg);
      if (options.hasOption(opt)) {
        tokens.add(arg);
      } else {
        if (options.hasOption(arg.substring(0, 2))) { // Buggy Line
          // the format is --foo=value or -foo=value
          // the format is a special properties option (-Dproperty=value) // Buggy Line
          tokens.add(arg.substring(0, 2)); // -D
          tokens.add(arg.substring(2)); // property=value
        } else {
          tokens.add(arg);
        }
      }
    } else {
      tokens.add(arg);
    }
  }
  return (String[]) tokens.toArray(new String[tokens.size()]);
}
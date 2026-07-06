public List getValues(final Option option,
                      List defaultValues) {
    // initialize the return list
    List valueList = (List) values.get(option);

    // grab the correct default values
    if ((valueList == null) || valueList.isEmpty()) {
        valueList = defaultValues;
    }

    // augment the list with the default values
    if ((valueList == null) || valueList.isEmpty()) {
        valueList = (List) this.defaultValues.get(option);
    } else {
        // copy the list first
        valueList = new ArrayList(valueList);
    }

    // if there are more default values as specified, add them to the list.
    return valueList == null ? Collections.EMPTY_LIST : valueList;
}
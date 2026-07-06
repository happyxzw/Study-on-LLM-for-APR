public StringBuffer format(Calendar calendar, StringBuffer buf) {
    if (mTimeZoneForced) {
        calendar = (Calendar) calendar.clone().clone();
        calendar.setTimeZone(mTimeZone);
    }
    return applyRules(calendar, buf);
}
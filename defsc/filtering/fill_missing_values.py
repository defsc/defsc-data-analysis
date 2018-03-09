
def simple_fill_missing_values(df):
    df = df.apply(lambda ts: ts.interpolate(method='nearest'))
    df = df.apply(lambda ts: ts.resample('1H').nearest())
    df = df.apply(lambda ts: ts.truncate(before='2017-09-23 23:59:00+00:00'))
    df = df.apply(lambda ts: ts.truncate(after='2018-02-24 23:59:00+00:00'))

    df = df.drop('wg-feelslike', axis=1)
    df = df.drop('wg-hum', axis=1)
    df = df.drop('wg-precip-1day', axis=1)
    df = df.drop('wg-precip-1h', axis=1)
    df = df.drop('wg-press', axis=1)
    df = df.drop('wg-tmp', axis=1)
    df = df.drop('wg-uv', axis=1)
    df = df.drop('wg-vis', axis=1)
    df = df.drop('wg-windchill', axis=1)
    df = df.drop('wg-wnd-gust', axis=1)
    df = df.drop('wg-wnd-spd', axis=1)

    return df

from pandas import DatetimeIndex, DataFrame, set_option, reset_option, isnull
import numpy as np

# depracated to delete
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
    df = df.drop('airly-pm1', axis=1)
    df = df.drop('airly-pm25', axis=1)
    df = df.drop('airly-press', axis=1)
    if 'here-traffic-speed' in df.columns:
        df = df.drop('here-traffic-speed', axis=1)
    df = df.drop('ow-hum', axis=1)
    #if 'ow-tmp' in df.columns:
    #    df = df.drop('ow-tmp', axis=1)

    return df

# depracated to delete
def fill_missing_values_with_truncate(df):
    df = df.apply(lambda ts: ts.interpolate(method='nearest'))
    df = df.apply(lambda ts: ts.resample('1H').nearest())
    df = df.apply(lambda ts: ts.truncate(before='2017-09-23 23:59:00+00:00'))
    df = df.apply(lambda ts: ts.truncate(after='2018-02-24 23:59:00+00:00'))

    return df

def drop_unnecessary_columns(df, column_to_retain):
    new_df =  df[column_to_retain]

    return new_df

def add_column_with_number_of_year(df, column_name='number_of_year'):
    df[column_name] = df.index.map(lambda timestamp: timestamp.dayofyear)

    return df


def remove_outliers(df, column_name, quantile=0.95):
    q = df[column_name].quantile(quantile)

    df[df[column_name] < q]

    return df

def fill_missing_values_with_nearest(df, columns_to_retain, start='2017-09-23 00:00:00', end='2018-02-24 00:00:00', drop_column_factor=0.3, verbose=0):
    df = df[columns_to_retain]
    dict_of_series = {}

    for column in df.columns:
        ts = df[column]
        ts = ts.dropna()

        if verbose != 0:
            print('Time series length before reindexing')
            print(ts.isnull().sum())
            print(len(ts))

        ts = ts.resample('1H').nearest()

        new_index = DatetimeIndex(start=start, end=end, freq='H')
        ts = ts.reindex(new_index)

        if ts.isnull().sum() < drop_column_factor * len(ts):
            dict_of_series[column] = ts

    new_df = DataFrame(dict_of_series, index=new_index)
    new_df = new_df.dropna()

    if verbose != 0:
        print('Final dataframe length')
        print(new_df.isnull().sum())
        print(new_df.shape[0])

    return new_df

def drop_missing_values(df, columns_to_retain, start='2017-09-23 00:00:00', end='2018-02-24 00:00:00', drop_column_factor=0.3, verbose=0):
    df = df[columns_to_retain]
    dict_of_series = {}

    for column in df.columns:
        ts = df[column]
        ts = ts.dropna()

        if verbose != 0:
            print('Time series length before reindexing')
            print(ts.isnull().sum())
            print(len(ts))

        ts = ts.resample('1H',  how = 'mean')

        new_index = DatetimeIndex(start=start, end=end, freq='H')
        ts = ts.reindex(new_index)

        if ts.isnull().sum() < drop_column_factor * len(ts):
            dict_of_series[column] = ts

    new_df = DataFrame(dict_of_series, index=new_index)
    new_df = new_df.dropna()

    if verbose != 0:
        print('Final dataframe length')
        print(new_df.isnull().sum())
        print(new_df.shape[0])

    return new_df

def find_predecessor(ts, index):
    i = index - 1
    while True:
        elem = ts.iloc[i]
        if ~ np.isnan(elem):
            return elem
        i = i - 1

def find_successor(ts, index):
    i = index + 1
    while True:
        elem = ts.iloc[i]
        if ~ np.isnan(elem):
            return elem
        i = i + 1

def fill_missing_values_using_mean_before_after(df, columns_to_retain, start='2017-09-23 00:00:00', end='2018-02-24 00:00:00', drop_column_factor=0.3, verbose=0):
    #Estimation of missing values in air pollution data using single imputation techniques - before - after mean method
    columns_to_retain = columns_to_retain
    df = df[columns_to_retain]
    dict_of_series = {}

    for column in df.columns:
        ts = df[column]
        ts = ts.dropna()

        if verbose != 0:
            print('Time series length before reindexing')
            print(ts.isnull().sum())
            print(len(ts))

        ts = ts.resample('1H',  how = 'mean')

        new_index = DatetimeIndex(start=start, end=end, freq='H')
        ts = ts.reindex(new_index)

        if ts.isnull().sum() < drop_column_factor * len(ts):
            dict_of_series[column] = ts

    new_df = DataFrame(dict_of_series, index=new_index)
    not_na_index = new_df[new_df.notna().all(axis=1)].index

    new_df = new_df.truncate(before=not_na_index[0], after=not_na_index[-1])

    for column in new_df.columns:
        ts = new_df[column]
        for i in range(ts.size):
            elem = ts.iloc[i]
            if np.isnan(elem):
                predecessor = find_predecessor(ts, i)
                successor = find_successor(ts, i)
                ts.iloc[i] = (predecessor + successor) / 2.0

    if verbose != 0:
        print('Final dataframe length')
        print(new_df.isnull().sum())
        print(new_df.shape[0])


    return new_df

from pandas import concat
from pandas import read_csv
from pandas import to_datetime


def transform_dataframe_to_supervised(old_df, x_column_names, y_column_names, number_of_timestep_ahead,
                                      number_of_timestep_backward):
    cols, names = list(), list()

    for lag in range(number_of_timestep_backward, 0, -1):
        for time_series_label in x_column_names:
            cols.append(old_df[time_series_label].shift(lag))
            names.append('{}(t-{})'.format(time_series_label, lag))

    for lag in range(0, number_of_timestep_ahead):
        for time_series_label in y_column_names:
            cols.append(old_df[time_series_label].shift(-lag))
            names.append('{}(t+{})'.format(time_series_label, lag))

    new_df = concat(cols, axis=1)
    new_df.columns = names

    return new_df


def split_timeseries_set_on_test_train(df_values, x_length, y_length, percantage_of_train_data=0.8):
    number_of_rows = df_values.shape[0]
    number_of_train_rows = int(number_of_rows * percantage_of_train_data)

    train_x = df_values[:number_of_train_rows, :x_length]
    train_y = df_values[:number_of_train_rows, -y_length:]

    test_x = df_values[number_of_train_rows:, :x_length]
    test_y = df_values[number_of_train_rows:, -y_length:]

    return (train_x, train_y, test_x, test_y)


if __name__ == "__main__":
    time_series_csv = '../data/raw-204.csv'
    df = read_csv(time_series_csv, header=0, index_col=0)
    df.index = to_datetime(df.index)

    print(df.head())

    x_column_names = ['airly-hum', 'airly-pm1', 'airly-pm10', 'airly-pm25', 'airly-press',
                      'airly-tmp', 'here-traffic-jam', 'here-traffic-speed', 'ow-hum',
                      'ow-press', 'ow-tmp', 'ow-vis', 'ow-wnd-deg', 'ow-wnd-spd',
                      'wg-feelslike', 'wg-hum', 'wg-precip-1day', 'wg-precip-1h', 'wg-press',
                      'wg-tmp', 'wg-uv', 'wg-vis', 'wg-windchill', 'wg-wnd-gust',
                      'wg-wnd-spd', 'wios-NO', 'wios-NO2']
    y_column_names = ['airly-pm1']

    df = transform_dataframe_to_supervised(df, x_column_names, y_column_names, 2, 3)

    print(df.head())

    df_values = split_timeseries_set_on_test_train(df.values, len(x_column_names) * 2, len(y_column_names) * 3)

    print(df_values[0].shape, df_values[1].shape, df_values[2].shape, df_values[3].shape)

import os

from pandas import read_csv, to_datetime

from defsc.data_structures_transformation.data_structures_transformation import transform_dataframe_to_supervised
from defsc.filtering.time_series_cleaning import drop_missing_values
from defsc.time_series_forecasting.forecasts import compare_methods

if __name__ == "__main__":
    msc_data_dir = os.environ['MSC_DATA']
    directory = os.path.join(msc_data_dir, 'multivariate-time-series-may-wios')

    train_period = 120 * 24
    number_of_models = 50

    for filename in os.listdir(directory):
        csv = os.path.join(directory, filename)
        df = read_csv(csv, header=0, index_col=0)
        df.index = to_datetime(df.index)

        y_column_names = ['airly-pm10']
        x_history_column_names = ['airly-pm10']
        x_forecast_column_names = ['ow-wnd-spd', 'ow-tmp', 'ow-hum']

        number_of_timestep_ahead = 24
        number_of_timestep_backward = 24

        if not all(column in df.columns for column in (x_history_column_names + x_forecast_column_names)):
            print(filename)
            print('Not all columns')
            continue

        df = drop_missing_values(df, x_history_column_names + x_forecast_column_names, start='2017-09-23 00:00:00', end='2018-04-30 23:00:00')

        if not all(column in df.columns for column in (x_history_column_names + x_forecast_column_names)):
            print(filename)
            print('Not all columns after dropping missing values')
            continue

        x_length = len(x_history_column_names) * number_of_timestep_backward + len(x_forecast_column_names) * number_of_timestep_ahead
        y_length = len(y_column_names) * number_of_timestep_ahead

        end_index = df.values.shape[0] - train_period - number_of_timestep_backward - number_of_timestep_ahead
        step = int(end_index / number_of_models)
        block_length = number_of_timestep_backward + train_period + number_of_timestep_backward + number_of_timestep_ahead

        for i in range(number_of_models):
            partial_df = df.iloc[step * i:step * i + block_length][:]

            new_df = transform_dataframe_to_supervised(partial_df, x_history_column_names, x_forecast_column_names, y_column_names,
                                                       number_of_timestep_ahead,
                                                       number_of_timestep_backward)

            new_df = new_df.dropna()

            if(new_df.values.shape[0]==0):
                continue

            number_of_rows = new_df.values.shape[0]
            number_of_train_rows = number_of_rows - 1

            train_x = new_df.values[:number_of_train_rows - number_of_timestep_backward, :x_length]
            train_y = new_df.values[:number_of_train_rows - number_of_timestep_backward, -y_length:]

            test_x = new_df.values[number_of_train_rows:, :x_length]
            test_y = new_df.values[number_of_train_rows:, -y_length:]

            id = os.path.splitext(filename)[0]  + '_' + partial_df.index[0].strftime('%Y-%m-%d') + '_' + partial_df.index[-1].strftime('%Y-%m-%d') + '_train_len:' + str(train_x.shape[0])

            compare_methods(new_df, train_x, train_y, test_x, test_y, number_of_timestep_ahead,
                                      number_of_timestep_backward, id, x_history_column_names + x_forecast_column_names, y_column_names)

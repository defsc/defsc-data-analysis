import math
import os

import itertools
from pandas import read_csv, to_datetime, TimeGrouper, Series, concat

from defsc.filtering.time_series_cleaning import simple_fill_missing_values, add_column_with_number_of_year, \
    fill_missing_values_with_truncate, drop_unnecessary_columns, remove_outliers, drop_missing_values
from defsc.time_series_forecasting.forecasts import perform_persistence_model_prediction, evaluate_method_results, \
    perform_arima_prediction, perform_linear_regression_prediction, perform_random_forest_regression_prediction, \
    perform_nn_lstm_prediction, perform_nn_mlp_prediction
from defsc.visualizations.time_series_visualization import plot_all_time_series_from_dataframe, \
    plot_all_time_series_decomposition_from_dataframe, plot_heat_map_of_correlation_coefficients, crosscorr, \
    scatter_plot_of_the_value_of_one_parameter_in_dependence_of_another
import statsmodels.api as sm


def compare_methods_each_iter(df, train_x, train_y, test_x, test_y, id, x_column_names, y_column_names):
    number_of_timestep_ahead = 1

    persistence_model_result = perform_persistence_model_prediction(df, y_column_names[0] + '(t-1)', len(test_y),
                                                                    number_of_timestep_ahead)
    evaluate_method_results(id + '_persistence-model-regression', test_y,
                            persistence_model_result)

    linear_regression_result = perform_linear_regression_prediction(df, train_x, train_y, test_x,
                                                                    number_of_timestep_ahead)
    evaluate_method_results(id + '_linear-regression', test_y,
                            linear_regression_result)

    random_forest_regression_result = perform_random_forest_regression_prediction(train_x, train_y, test_x,
                                                                                  number_of_timestep_ahead)
    evaluate_method_results(id + '_radnom-forest-regression',
                            test_y,
                            random_forest_regression_result)

    nn_lstm_regression_result = perform_nn_lstm_prediction(train_x, train_y, test_x, test_y,
                                                           number_of_timestep_ahead, 10,
                                                           train_x.shape[1])
    evaluate_method_results(id + '_nn-lstm-regression', test_y,
                            nn_lstm_regression_result)

    nn_mlp_regression_result = perform_nn_mlp_prediction(train_x, train_y, test_x, test_y, number_of_timestep_ahead)
    evaluate_method_results(id + '_nn-mlp-regression', test_y,
                            nn_mlp_regression_result)

def transform_dataframe_to_supervised(old_df, x_history_column_names, x_forecast_column_names, y_column_names):
    cols, names = list(), list()

    # x_hisotry
    lag = 1
    cols.append(old_df['airly-pm10'].shift(lag))
    names.append('{}(t-{})'.format('airly-pm10', lag))

    # x_forecast
    lag = 23
    cols.append(old_df['ow-wnd-spd'].shift(-lag))
    names.append('{}(t+{})'.format('ow-wnd-spd', lag))

    lag = 23
    cols.append(old_df['ow-tmp'].shift(-lag))
    names.append('{}(t+{})'.format('ow-tmp', lag))

    lag = 23
    cols.append(old_df['ow-hum'].shift(-lag))
    names.append('{}(t+{})'.format('ow-hum', lag))

    lag = 23
    cols.append(old_df['ow-press'].shift(-lag))
    names.append('{}(t+{})'.format('ow-press', lag))

    # y
    lag = 24
    cols.append(old_df['airly-pm10'].shift(-lag))
    names.append('{}(t+{})'.format('airly-pm10', lag))

    new_df = concat(cols, axis=1)
    new_df.columns = names

    return new_df


if __name__ == "__main__":
    directory = './data/multivariate-time-series'

    train_period = 15 * 24
    number_of_models = 20

    for filename in os.listdir(directory):
        if filename == 'pollution.csv':
            continue

        csv = os.path.join(directory, filename)
        df = read_csv(csv, header=0, index_col=0)
        df.index = to_datetime(df.index)

        if 'ow-wnd-spd' in df.columns and 'ow-wnd-deg' in df.columns:
            df['ow-wnd-x'] = df.apply(lambda row: row['ow-wnd-spd'] * math.cos(math.radians(row['ow-wnd-deg'])), axis=1)
            df['ow-wnd-y'] = df.apply(lambda row: row['ow-wnd-spd'] * math.sin(math.radians(row['ow-wnd-deg'])), axis=1)

        y_column_names = ['airly-pm10']
        #x_column_names = ['airly-pm1', 'ow-wnd-x', 'here-traffic-jam', 'airly-tmp', 'ow-wnd-y', 'ow-press']
        x_history_column_names = ['airly-pm10']
        x_forecast_column_names = ['ow-wnd-spd','ow-tmp','ow-hum','ow-press']
        #x_forecast_column_names = ['airly-tmp','ow-wnd-x','ow-wnd-y','here-traffic-jam','airly-hum','airly-press']

        number_of_timestep_ahead = 24
        number_of_timestep_backward = 24

        if not all(column in df.columns for column in (x_history_column_names + x_forecast_column_names)):
            print(filename)
            print('Not all columns')
            continue

        df = drop_missing_values(df, x_history_column_names + x_forecast_column_names, start='2017-09-23 00:00:00', end='2018-02-24 00:00:00')

        if not all(column in df.columns for column in (x_history_column_names + x_forecast_column_names)):
            print(filename)
            print('Not all columns after dropping missing values')
            continue

        x_length = 5
        y_length = 1

        end_index = df.values.shape[0] - train_period - number_of_timestep_backward - number_of_timestep_ahead
        step = int(end_index / number_of_models)
        block_length = number_of_timestep_backward + train_period + number_of_timestep_backward + number_of_timestep_ahead

        for i in range(number_of_models):
            partial_df = df.iloc[step * i:step * i + block_length][:]

            new_df = transform_dataframe_to_supervised(partial_df, x_history_column_names, x_forecast_column_names, y_column_names)

            new_df = new_df.dropna()

            number_of_rows = new_df.values.shape[0]
            number_of_train_rows = number_of_rows - 1

            train_x = new_df.values[:number_of_train_rows - number_of_timestep_backward, :x_length]
            train_y = new_df.values[:number_of_train_rows - number_of_timestep_backward, -y_length:]

            test_x = new_df.values[number_of_train_rows:, :x_length]
            test_y = new_df.values[number_of_train_rows:, -y_length:]

            if(train_y.shape[0]==0 or test_y.shape[0]==0):
                continue

            id = os.path.splitext(filename)[0]  + '_' + partial_df.index[0].strftime('%Y-%m-%d') + '_' + partial_df.index[-1].strftime('%Y-%m-%d') + '_train_len:' + str(train_x.shape[0])

            compare_methods_each_iter(new_df, train_x, train_y, test_x, test_y, id, x_history_column_names + x_forecast_column_names, y_column_names)
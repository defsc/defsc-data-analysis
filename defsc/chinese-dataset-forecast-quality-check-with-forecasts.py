import math
import os
import numpy as np

import itertools

import sys
from pandas import read_csv, to_datetime, TimeGrouper, Series, concat

from defsc.data_structures_transformation.data_structures_transformation import transform_dataframe_to_supervised
from defsc.filtering.time_series_cleaning import simple_fill_missing_values, add_column_with_number_of_year, \
    fill_missing_values_with_truncate, drop_unnecessary_columns, drop_missing_values
from defsc.time_series_forecasting.forecasts import perform_persistence_model_prediction, evaluate_method_results, \
    perform_arima_prediction, perform_linear_regression_prediction, perform_random_forest_regression_prediction, \
    perform_nn_lstm_prediction, perform_nn_mlp_prediction
from defsc.visualizations.time_series_visualization import plot_all_time_series_from_dataframe, \
    plot_all_time_series_decomposition_from_dataframe, plot_heat_map_of_correlation_coefficients, crosscorr, \
    scatter_plot_of_the_value_of_one_parameter_in_dependence_of_another
import statsmodels.api as sm


def compare_methods_each_iter(df, train_x, train_y, test_x, test_y, number_of_timestep_ahead,
                              number_of_timestep_backward, id, x_column_names, y_column_names):
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
                                                           number_of_timestep_ahead, number_of_timestep_backward,
                                                           train_x.shape[1])
    evaluate_method_results(id + '_nn-lstm-regression', test_y,
                            nn_lstm_regression_result)

    nn_mlp_regression_result = perform_nn_mlp_prediction(train_x, train_y, test_x, test_y, number_of_timestep_ahead)
    evaluate_method_results(id + '_nn-mlp-regression', test_y,
                            nn_mlp_regression_result)


if __name__ == "__main__":
    csv = './data/multivariate-time-series/pollution.csv'

    train_period = 3 * 365 * 24
    number_of_models = 50

    df = read_csv(csv, header=0, index_col=0)
    df.index = to_datetime(df.index)

    y_column_names = ['pollution']
    #x_column_names = ['airly-pm1', 'ow-wnd-x', 'here-traffic-jam', 'airly-tmp', 'ow-wnd-y', 'ow-press']
    x_history_column_names = ['pollution']
    x_forecast_column_names = ['dew', 'temp', 'wnd_spd', 'snow','rain','press']
    #x_forecast_column_names = ['airly-tmp','ow-wnd-x','ow-wnd-y','here-traffic-jam','airly-hum','airly-press']

    number_of_timestep_ahead = 24
    number_of_timestep_backward = 24

    if not all(column in df.columns for column in (x_history_column_names + x_forecast_column_names)):
        print(csv)
        print('Not all columns')
        sys.exit(1)

    df = drop_missing_values(df, x_history_column_names + x_forecast_column_names, start='2010-01-02 00:00:00', end='2014-12-31 23:00:00')

    if not all(column in df.columns for column in (x_history_column_names + x_forecast_column_names)):
        print(csv)
        print('Not all columns after dropping missing values')
        sys.exit(1)

    x_length = len(x_history_column_names) * number_of_timestep_backward + len(x_forecast_column_names) * number_of_timestep_ahead
    y_length = len(y_column_names) * number_of_timestep_ahead

    end_index = df.values.shape[0] - train_period - number_of_timestep_backward - number_of_timestep_ahead
    step = int(end_index / number_of_models)
    block_length = number_of_timestep_backward + train_period + number_of_timestep_backward + number_of_timestep_ahead

    for i in range(number_of_models):
        partial_df = df.iloc[step * i:step * i + block_length][:]

        #from matplotlib import pyplot
        #partial_df.plot(subplots=True)
        #pyplot.show()

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

        id = 'pollution.csv'  + '_' + partial_df.index[0].strftime('%Y-%m-%d') + '_' + partial_df.index[-1].strftime('%Y-%m-%d') + '_train_len:' + str(train_x.shape[0])

        compare_methods_each_iter(new_df, train_x, train_y, test_x, test_y, number_of_timestep_ahead,
                                  number_of_timestep_backward, id, x_history_column_names + x_forecast_column_names, y_column_names)

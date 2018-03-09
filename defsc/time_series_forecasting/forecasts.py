from defsc.data_structures_transformation.data_structures_transformation import transform_dataframe_to_supervised
from defsc.data_structures_transformation.data_structures_transformation import split_timeseries_set_on_test_train
from defsc.time_series_forecasting.linear_regression import generate_linear_regression_model
from defsc.time_series_forecasting.random_forest_regression_tree import generate_random_forest_regression_tree_model
from defsc.filtering.fill_missing_values import simple_fill_missing_values
from defsc.time_series_forecasting.nn_lstm_forecasting import generate_nn_lstm_model
from defsc.time_series_forecasting.nn_lstm_forecasting import reshape_input_for_lstm
from defsc.visualizations.time_series_visualization import plot_histograms_of_forecasts_errors_per_hour
from defsc.utils.utils import print_number_of_nan_values

import matplotlib.pyplot as pyplot

import numpy as np
from math import sqrt
from pandas import to_datetime
from pandas import read_csv
from sklearn.metrics import mean_squared_error


def forecast_linear(model, X):
    forecast = model.predict(X)
    return forecast.flatten()


def make_forecasts(model, test, n_ahead):
    forecasts_y = list()

    for i in range(len(test)):
        x = test[i, :]
        forecast = forecast_linear(model, x)
        forecasts_y.append(forecast)

    return forecasts_y


def calculate_rsme_metrics(y_real, y_predicted):
    for hour in range(y_real.shape[1]):
        rmse = sqrt(mean_squared_error(y_real[:, hour], np.asarray(y_predicted)[:, hour]))
        print('t+%d RMSE: %f' % ((hour + 1), rmse))


if __name__ == "__main__":
    time_series_csv = '../data/raw-204.csv'
    df = read_csv(time_series_csv, header=0, index_col=0)
    df.index = to_datetime(df.index)

    df = simple_fill_missing_values(df)

    number_of_timestep_ahead = 24
    number_of_timestep_backward = 24

    x_column_names = ['airly-hum', 'airly-pm1', 'airly-pm10', 'airly-pm25', 'airly-press',
                      'airly-tmp', 'here-traffic-jam', 'here-traffic-speed', 'ow-hum',
                      'ow-press', 'ow-tmp', 'ow-vis', 'ow-wnd-deg', 'ow-wnd-spd', 'wios-NO', 'wios-NO2']
    y_column_names = ['wios-NO']

    df = transform_dataframe_to_supervised(df, x_column_names, y_column_names, number_of_timestep_ahead,
                                           number_of_timestep_backward)

    df = df.dropna()

    train_x, train_y, test_x, test_y = split_timeseries_set_on_test_train(df.values,
                                                                          len(
                                                                              x_column_names) * number_of_timestep_backward,
                                                                          len(
                                                                              y_column_names) * number_of_timestep_ahead)

    # train_x = reshape_input_for_lstm(train_x, number_of_timestep_backward, len(x_column_names))
    # test_x = reshape_input_for_lstm(test_x, number_of_timestep_backward, len(x_column_names))

    # model = generate_nn_lstm_model(train_x, train_y, test_x, test_y, number_of_timestep_ahead)

    model = generate_linear_regression_model(train_x, train_y)
    linear_regression_forecast_y = make_forecasts(model, test_x, number_of_timestep_ahead)
    calculate_rsme_metrics(test_y, linear_regression_forecast_y)
    plot_histograms_of_forecasts_errors_per_hour(test_y, linear_regression_forecast_y)

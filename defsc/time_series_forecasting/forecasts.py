import matplotlib
matplotlib.use('Qt5Agg')

import time

from defsc.time_series_forecasting.arima import generate_arima_forecast
from defsc.time_series_forecasting.nn_mlp_forecasting import generate_nn_mlp_model
from defsc.data_structures_transformation.data_structures_transformation import transform_dataframe_to_supervised
from defsc.data_structures_transformation.data_structures_transformation import split_timeseries_set_on_test_train
from defsc.time_series_forecasting.linear_regression import generate_linear_regression_model
from defsc.time_series_forecasting.random_forest_regression_tree import generate_random_forest_regression_tree_model
# from defsc.time_series_forecasting.random_forest_regression_tree import generate_random_forest_regression_tree_model
from defsc.filtering.fill_missing_values import simple_fill_missing_values
from defsc.time_series_forecasting.nn_lstm_forecasting import generate_nn_lstm_model
from defsc.time_series_forecasting.nn_lstm_forecasting import reshape_input_for_lstm
from defsc.visualizations.time_series_visualization import plot_histograms_of_forecasts_errors_per_hour
from defsc.visualizations.time_series_visualization import plot_forecasting_result
from defsc.visualizations.time_series_visualization import plot_forecasting_result_v2
from defsc.visualizations.time_series_visualization import plot_forecast_result_in_3d
from defsc.visualizations.time_series_visualization import plot_forecast_result_as_heat_map
from defsc.utils.utils import print_number_of_nan_values
from defsc.filtering.time_series_filtering import testGauss
from mpl_toolkits.mplot3d import Axes3D
from pandas import Series

import matplotlib.pyplot as pyplot

import numpy as np
import os
from math import sqrt
from pandas import to_datetime
from pandas import read_csv
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def forecast_linear(model, X):
    X = X.reshape(1,-1)
    forecast = model.predict(X)
    return forecast.flatten()


def make_forecasts(model, test, n_ahead):
    forecasts_y = list()

    for i in range(len(test)):
        x = test[i, :]
        forecast = forecast_linear(model, x)
        forecasts_y.append(forecast)

    return np.asarray(forecasts_y).clip(min=0)


def calculate_rsme_metrics(y_real, y_predicted):
    rmse = []
    for hour in range(y_real.shape[1]):
        rmse.append(sqrt(mean_squared_error(y_real[:, hour], y_predicted[:, hour])))
    rmse = list(map(lambda x: format(x, '.2f'), rmse))
    print(rmse)


def calculate_mean_absolute_metrics(y_real, y_predicted):
    rmse = []
    for hour in range(y_real.shape[1]):
        rmse.append(mean_absolute_error(y_real[:, hour], y_predicted[:, hour]))
    rmse = list(map(lambda x: format(x, '.2f'), rmse))
    print(rmse)

def update_df_column(df, column_name, update_fn):
    new_column = Series(update_fn(df[column_name].values, len(df[column_name].values)), name=column_name, index=df.index)
    df.update(new_column)

def save_prediction_result_to_file(y_real, y_predicted, filename_base=None):
    if filename_base == None:
        timestamp = time.strftime("%Y-%m-%d-%H-%M")
        filename_base = timestamp

    np.savetxt(filename_base + 'y_real' + '.txt', y_real, delimiter=',', fmt='%.2f')
    np.savetxt(filename_base + 'y_predicted' + '.txt', y_predicted, delimiter=',', fmt='%.2f')

def perform_arima_prediction(df, predicted_ts_name, number_of_timestep_ahead):
    #'airly-pm1(t+0)'
    arima_result = generate_arima_forecast(df[predicted_ts_name], number_of_timestep_ahead)

    return arima_result


def perform_linear_regression_prediction():
    model = generate_linear_regression_model(train_x, train_y)
    forecast = make_forecasts(model, test_x, number_of_timestep_ahead)

    return forecast

def perform_random_forest_regression_prediction():
    model = generate_random_forest_regression_tree_model(train_x, train_y, test_x, test_y, number_of_timestep_ahead)
    forecast = make_forecasts(model, test_x, number_of_timestep_ahead)

    return forecast

def perform_nn_mlp_prediction(train_x, train_y, test_x, test_y):
    model = generate_nn_mlp_model(train_x, train_y, test_x, test_y, number_of_timestep_ahead)
    forecast = make_forecasts(model, test_x, number_of_timestep_ahead)

    return forecast

def perform_nn_lstm_prediction(train_x, train_y, test_x, test_y):
    train_x = reshape_input_for_lstm(train_x, number_of_timestep_backward, len(x_column_names))
    test_x = reshape_input_for_lstm(test_x, number_of_timestep_backward, len(x_column_names))

    model = generate_nn_lstm_model(train_x, train_y, test_x, test_y, number_of_timestep_ahead)
    forecast = np.asarray(model.predict(test_x)).clip(0)

    return forecast

def evaluate_method_results(id, y_real, y_predicted):
    calculate_mean_absolute_metrics(y_real, y_predicted)
    plot_histograms_of_forecasts_errors_per_hour(y_real, y_predicted)
    plot_forecast_result_as_heat_map(y_real, y_predicted)
    plot_forecasting_result_v2(y_real, y_predicted)

if __name__ == "__main__":
    directory = '../data'
    for filename in os.listdir(directory):
        print(filename)
        if filename == 'pollution.csv' or filename == 'raw-626.csv':
            continue
        csv = os.path.join(directory, filename)
        df = read_csv(csv, header=0, index_col=0)
        df.index = to_datetime(df.index)

        df = simple_fill_missing_values(df)

        number_of_timestep_ahead = 24
        number_of_timestep_backward = 24

        x_column_names = df.columns
        y_column_names = ['airly-pm1']

        df = transform_dataframe_to_supervised(df, x_column_names, y_column_names, number_of_timestep_ahead,
                                               number_of_timestep_backward)

        df = df.dropna()

        train_x, train_y, test_x, test_y = split_timeseries_set_on_test_train(df.values,
                                                                              len(
                                                                                  x_column_names) * number_of_timestep_backward,
                                                                              len(
                                                                                  y_column_names) * number_of_timestep_ahead)


        arima_result = perform_arima_prediction(df, 'airly-pm1(t+0)', number_of_timestep_ahead)
        evaluate_method_results('arima', test_y, arima_result)

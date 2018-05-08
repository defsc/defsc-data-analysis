import warnings

from sklearn.preprocessing import MinMaxScaler

from defsc.time_series_forecasting.svr import generate_svr_regression_model

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib

matplotlib.use('Qt5Agg')

import time

from defsc.time_series_forecasting.arima import generate_arima_forecast
from defsc.time_series_forecasting.nn_mlp_forecasting import generate_nn_mlp_model
from defsc.data_structures_transformation.data_structures_transformation import transform_dataframe_to_supervised
from defsc.data_structures_transformation.data_structures_transformation import split_timeseries_set_on_test_train
from defsc.time_series_forecasting.linear_regression import generate_linear_regression_model
from defsc.time_series_forecasting.random_forest_regression_tree import generate_random_forest_regression_tree_model
from defsc.filtering.time_series_cleaning import simple_fill_missing_values
from defsc.time_series_forecasting.nn_lstm_forecasting import generate_nn_lstm_model
from defsc.time_series_forecasting.nn_lstm_forecasting import reshape_input_for_lstm
from defsc.visualizations.time_series_visualization import plot_histograms_of_forecasts_errors_per_hour, \
    plot_timeseries, plot_all_time_series_from_dataframe
from defsc.visualizations.time_series_visualization import plot_forecasting_result
from defsc.visualizations.time_series_visualization import plot_forecasting_result_v2
from defsc.visualizations.time_series_visualization import plot_forecast_result_in_3d
from defsc.visualizations.time_series_visualization import plot_forecast_result_as_heat_map
from defsc.utils.utils import print_number_of_nan_values
from defsc.filtering.time_series_smoothing import testGauss
from pandas import Series, concat, DataFrame

import matplotlib.pyplot as pyplot
import itertools
import numpy as np
import os
from math import sqrt
from pandas import to_datetime
from pandas import read_csv
from sklearn.metrics import mean_squared_error, mean_absolute_error

def forecast_linear(model, X):
    X = X.reshape(1, -1)
    forecast = model.predict(X)
    return forecast.flatten()


def make_forecasts(model, test, n_ahead):
    forecasts_y = list()

    for i in range(len(test)):
        x = test[i, :]
        forecast = forecast_linear(model, x)
        forecasts_y.append(forecast)

    #return np.asarray(forecasts_y).clip(min=0)
    return np.asarray(forecasts_y)


def calculate_rmse_metrics(y_real, y_predicted):
    rmse = []
    for hour in range(y_real.shape[1]):
        rmse.append(sqrt(mean_squared_error(y_real[:, hour], y_predicted[:, hour])))

    return  rmse

def calculate_nrmse_metrics(y_real, y_predicted):
    rmse = []
    for hour in range(y_real.shape[1]):
        rmse.append(sqrt(mean_squared_error(y_real[:, hour], y_predicted[:, hour])) / (np.max(y_real) - np.min(y_real)))

    return  rmse

def calculate_mae_metrics(y_real, y_predicted):
    mae = []
    for hour in range(y_real.shape[1]):
        mae.append(mean_absolute_error(y_real[:, hour], y_predicted[:, hour]))

    return mae


def update_df_column(df, column_name, update_fn):
    new_column = Series(update_fn(df[column_name].values, len(df[column_name].values)), name=column_name,
                        index=df.index)
    df.update(new_column)


def save_prediction_result_to_file(y_real, y_predicted, filename_base=None):
    if filename_base == None:
        timestamp = time.strftime("%Y-%m-%d-%H-%M")
        filename_base = timestamp

    np.savetxt(filename_base + 'y_real' + '.txt', y_real, delimiter=',', fmt='%.2f')
    np.savetxt(filename_base + 'y_predicted' + '.txt', y_predicted, delimiter=',', fmt='%.2f')


def perform_arima_prediction(df, predicted_ts_name, number_of_timestep_ahead,p=7,d=0,q=4):
    # 'airly-pm1(t+0)'
    arima_result = generate_arima_forecast(df[predicted_ts_name], number_of_timestep_ahead,p=p, d=d, q=q)

    return arima_result


def perform_linear_regression_prediction(df, train_x, train_y, test_x, number_of_timestep_ahead, print_coefficients=False):
    model = generate_linear_regression_model(train_x, train_y)

    if print_coefficients:
        coefficients = concat([DataFrame(df.columns[:-number_of_timestep_ahead]), DataFrame(np.transpose(model.estimators_[0].coef_))], axis=1)
        print(coefficients)

    forecast = make_forecasts(model, test_x, number_of_timestep_ahead)

    return forecast

def perform_svr_regression_prediction(train_x, train_y, test_x, number_of_timestep_ahead):
    model = generate_svr_regression_model(train_x, train_y)
    forecast = make_forecasts(model, test_x, number_of_timestep_ahead)

    return forecast

def perform_random_forest_regression_prediction(train_x, train_y, test_x, number_of_timestep_ahead):
    model = generate_random_forest_regression_tree_model(train_x, train_y)
    forecast = make_forecasts(model, test_x, number_of_timestep_ahead)

    return forecast


def perform_nn_mlp_prediction(train_x, train_y, test_x, test_y, number_of_timestep_ahead):
    train_x_scaler = MinMaxScaler(feature_range=(0,1))
    train_y_scaler = MinMaxScaler(feature_range=(0,1))

    train_x = train_x_scaler.fit_transform(train_x)
    train_y = train_y_scaler.fit_transform(train_y)
    test_x = train_x_scaler.transform(test_x)
    test_y = train_y_scaler.transform(test_y)

    model = generate_nn_mlp_model(train_x, train_y, test_x, test_y, number_of_timestep_ahead, verbose=0)
    forecast = make_forecasts(model, test_x, number_of_timestep_ahead)

    forecast = train_y_scaler.inverse_transform(forecast)

    return forecast


def perform_nn_lstm_prediction(train_x, train_y, test_x, test_y, number_of_timestep_ahead, number_of_timestep_backward,
                               number_of_input_parameters):
    train_x_scaler = MinMaxScaler(feature_range=(0,1))
    train_y_scaler = MinMaxScaler(feature_range=(0,1))

    train_x = train_x_scaler.fit_transform(train_x)
    train_y = train_y_scaler.fit_transform(train_y)
    test_x = train_x_scaler.transform(test_x)
    test_y = train_y_scaler.transform(test_y)

    train_x = reshape_input_for_lstm(train_x, number_of_timestep_ahead, number_of_input_parameters)
    test_x = reshape_input_for_lstm(test_x, number_of_timestep_ahead, number_of_input_parameters)

    model = generate_nn_lstm_model(train_x, train_y, test_x, test_y, number_of_timestep_ahead, verbose=0)
    forecast = np.asarray(model.predict(test_x)).clip(0)

    forecast = train_y_scaler.inverse_transform(forecast)

    return forecast


def perform_persistence_model_prediction(df, last_seen_column_name, number_of_test_y_row, number_of_timestep_ahead):
    forecast = []
    for last_seen_y in df[last_seen_column_name][-number_of_test_y_row:]:
        forecast.append(number_of_timestep_ahead * [last_seen_y])

    return np.asarray(forecast)


def perform_persistence_model_prediction_24(df, predicted_column_name, number_of_test_y_row, number_of_timestep_ahead):
    forecast = []

    for y_row in range(number_of_test_y_row):
        forecast_for_one_step = []
        for hour in range(number_of_timestep_ahead):
            column_name = predicted_column_name + '(t-{})'.format(hour+1)
            forecast_for_one_step.append(df[column_name][y_row])
        forecast.append(list(reversed(forecast_for_one_step)))

    return np.asarray(forecast)

def evaluate_method_results(id, y_real, y_predicted):
    mae = calculate_mae_metrics(y_real, y_predicted)
    rmse = calculate_rmse_metrics(y_real, y_predicted)
    nrmse = calculate_nrmse_metrics(y_real, y_predicted)
    print('{},{:.2f},{:.2f},{:.2f}'.format(id, np.average(mae), np.average(rmse), np.average(nrmse)))

    #plot_histograms_of_forecasts_errors_per_hour(y_real, y_predicted, save_to_file=True, filename=id)
    #plot_forecast_result_as_heat_map(y_real, y_predicted, save_to_file=True, filename=id)
    #plot_forecasting_result_v2(y_real, y_predicted, save_to_file=True, filename=id)
    plot_forecasting_result(y_real, y_predicted, save_to_file=True, filename=id)

def compare_methods(df, train_x, train_y, test_x, test_y, number_of_timestep_ahead, number_of_timestep_backward, filename, x_column_names, y_column_name):
    #plot_timeseries(df['airly-pm1(t+0)'], save_to_file=True,
    #                filename='forecasted_timeseries_' + os.path.splitext(filename)[0])

    persistence_model_result = perform_persistence_model_prediction(df, y_column_name + '(t-1)', len(test_y),
                                                                    number_of_timestep_ahead)
    evaluate_method_results('persistence-model-regression_' + os.path.splitext(filename)[0], test_y,
                            persistence_model_result)

    #persistence_model_result = perform_persistence_model_prediction_24(df, y_column_name, len(test_y),
    #                                                                number_of_timestep_ahead)
    #evaluate_method_results('persistence-model-24-regression_' + os.path.splitext(filename)[0], test_y,
    #                        persistence_model_result)

    linear_regression_result = perform_linear_regression_prediction(df, train_x, train_y, test_x,
                                                                    number_of_timestep_ahead)
    evaluate_method_results('_'.join(x_column_names) + '_linear-regression_' + os.path.splitext(filename)[0], test_y, linear_regression_result)

    #arima_result = perform_arima_prediction(df, y_column_name + '(t+0)', number_of_timestep_ahead)
    #evaluate_method_results('_'.join(x_column_names) + '_arima_' + os.path.splitext(filename)[0], test_y, arima_result)

    #svr_regression_result = perform_svr_regression_prediction(train_x, train_y, test_x,
    #                                                                number_of_timestep_ahead)
    #evaluate_method_results('_'.join(x_column_names) + 'svr-regression_' + os.path.splitext(filename)[0], test_y, svr_regression_result)

    #random_forest_regression_result = perform_random_forest_regression_prediction(train_x, train_y, test_x,
    #                                                                              number_of_timestep_ahead)
    #evaluate_method_results('_'.join(x_column_names) + '_radnom-forest-regression_' + os.path.splitext(filename)[0], test_y,
    #                        random_forest_regression_result)

    # nn_lstm_regression_result = perform_nn_lstm_prediction(train_x, train_y, test_x, test_y,
    #                                                        number_of_timestep_ahead, number_of_timestep_backward,
    #                                                        train_x.shape[1])
    #evaluate_method_results('_'.join(x_column_names) + '_nn-lstm-regression_' + os.path.splitext(filename)[0], test_y, nn_lstm_regression_result)

    #nn_mlp_regression_result = perform_nn_mlp_prediction(train_x, train_y, test_x, test_y, number_of_timestep_ahead)
    #evaluate_method_results('_'.join(x_column_names) + '_nn-mlp-regression_' + os.path.splitext(filename)[0], test_y, nn_mlp_regression_result)


import os

import itertools
from pandas import read_csv, to_datetime

from defsc.data_structures_transformation.data_structures_transformation import transform_dataframe_to_supervised, \
    split_timeseries_set_on_test_train
from defsc.filtering.fill_missing_values import simple_fill_missing_values, fill_missing_values_with_truncate, \
    drop_unnecessary_columns, add_column_with_number_of_year
from defsc.time_series_forecasting.forecasts import perform_persistence_model_prediction, evaluate_method_results, \
    perform_arima_prediction, perform_linear_regression_prediction, perform_random_forest_regression_prediction, \
    perform_nn_lstm_prediction, perform_nn_mlp_prediction
from defsc.visualizations.time_series_visualization import plot_all_time_series_from_dataframe, \
    plot_all_time_series_decomposition_from_dataframe
import statsmodels.api as sm


def compare_methods_once(df, train_x, train_y, test_x, test_y, number_of_timestep_ahead, number_of_timestep_backward, filename, x_column_names, y_column_names):
    persistence_model_result = perform_persistence_model_prediction(df, y_column_names[0] + '(t-1)', len(test_y),
                                                                    number_of_timestep_ahead)
    evaluate_method_results('persistence-model-regression_' + os.path.splitext(filename)[0], test_y,
                            persistence_model_result)

    #arima_result = perform_arima_prediction(df, 'airly-pm1(t+0)', number_of_timestep_ahead)
    #evaluate_method_results('_'.join(x_column_names) + '_arima_' + os.path.splitext(filename)[0], test_y, arima_result)

def compare_methods_each_iter(df, train_x, train_y, test_x, test_y, number_of_timestep_ahead, number_of_timestep_backward, filename, x_column_names):
    linear_regression_result = perform_linear_regression_prediction(train_x, train_y, test_x,
                                                                    number_of_timestep_ahead)
    evaluate_method_results('_'.join(x_column_names) + '_linear-regression_' + os.path.splitext(filename)[0], test_y, linear_regression_result)

    random_forest_regression_result = perform_random_forest_regression_prediction(train_x, train_y, test_x,
                                                                                  number_of_timestep_ahead)
    evaluate_method_results('_'.join(x_column_names) + '_radnom-forest-regression_' + os.path.splitext(filename)[0], test_y,
                            random_forest_regression_result)

    nn_lstm_regression_result = perform_nn_lstm_prediction(train_x, train_y, test_x, test_y,
                                                           number_of_timestep_ahead, number_of_timestep_backward,
                                                           len(x_column_names))
    evaluate_method_results('_'.join(x_column_names) + '_nn-lstm-regression_' + os.path.splitext(filename)[0], test_y, nn_lstm_regression_result)

    nn_mlp_regression_result = perform_nn_mlp_prediction(train_x, train_y, test_x, test_y, number_of_timestep_ahead)
    evaluate_method_results('_'.join(x_column_names) + '_nn-mlp-regression_' + os.path.splitext(filename)[0], test_y, nn_mlp_regression_result)



if __name__ == "__main__":
    directory = './data/multivariate-time-series'
    for filename in os.listdir(directory):

        if filename == 'pollution.csv':
            continue

        if filename not in ['raw-559.csv', 'raw-811.csv', 'raw-181.csv', 'raw-221.csv', 'raw-895.csv', 'raw-396.csv']:
            continue

        csv = os.path.join(directory, filename)
        df = read_csv(csv, header=0, index_col=0)
        df.index = to_datetime(df.index)

        number_of_timestep_ahead = 24
        number_of_timestep_backward = 24

        forecast_params= ['', 'ow-wnd-spd','airly-hum', 'airly-press', 'airly-tmp', 'ow-hum', 'ow-press', 'ow-tmp' ,'wg-feelslike', 'wg-hum','wg-precip-1h', 'wg-press', 'wg-tmp', 'wg-uv','wg-windchill', 'wg-wnd-gust']

        for forecast_param in forecast_params:
            print(forecast_param)

            y_column_names = ['airly-pm10']
            x_history_column_names = ['airly-pm10', 'number_of_year']
            if forecast_params[0] == '':
                x_forecast_column_names = []
            else:
                x_forecast_column_names = [forecast_param]

            df = fill_missing_values_with_truncate(df)
            df = drop_unnecessary_columns(df, x_history_column_names + x_forecast_column_names)

            if not all(column in df.columns for column in (x_history_column_names + x_forecast_column_names)):
                continue

            new_df = transform_dataframe_to_supervised(df, x_history_column_names, x_forecast_column_names, y_column_names, number_of_timestep_ahead,
                                                   number_of_timestep_backward)

            new_df = new_df.dropna()


            train_x, train_y, test_x, test_y = split_timeseries_set_on_test_train(new_df.values,
                                                                                  len(
                                                                                    x_history_column_names) * number_of_timestep_backward +
                                                                                  len(
                                                                                    x_forecast_column_names) * number_of_timestep_ahead,
                                                                                  len(
                                                                                      y_column_names) * number_of_timestep_ahead,
                                                                                  number_of_timestep_ahead)

            compare_methods_once(new_df, train_x, train_y, test_x, test_y, number_of_timestep_ahead,
                                 number_of_timestep_backward, filename, x_history_column_names + x_forecast_column_names, y_column_names)

            compare_methods_each_iter(new_df, train_x, train_y, test_x, test_y, number_of_timestep_ahead,
                                      number_of_timestep_backward, filename, x_history_column_names + x_forecast_column_names)

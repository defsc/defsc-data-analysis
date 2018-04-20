import math
import os

import itertools
from pandas import read_csv, to_datetime, TimeGrouper, Series

from defsc.data_structures_transformation.data_structures_transformation import transform_dataframe_to_supervised, \
    split_timeseries_set_on_test_train
from defsc.filtering.fill_missing_values import simple_fill_missing_values
from defsc.time_series_forecasting.forecasts import perform_persistence_model_prediction, evaluate_method_results, \
    perform_arima_prediction, perform_linear_regression_prediction, perform_random_forest_regression_prediction, \
    perform_nn_lstm_prediction, perform_nn_mlp_prediction
from defsc.visualizations.time_series_visualization import plot_all_time_series_from_dataframe, \
    plot_all_time_series_decomposition_from_dataframe, plot_heat_map_of_correlation_coefficients, crosscorr, \
    scatter_plot_of_the_value_of_one_parameter_in_dependence_of_another
import statsmodels.api as sm

def compare_methods_once(df, train_x, train_y, test_x, test_y, number_of_timestep_ahead, number_of_timestep_backward, filename, x_column_names):
    persistence_model_result = perform_persistence_model_prediction(df, 'airly-pm1(t-1)', len(test_y),
                                                                    number_of_timestep_ahead)
    evaluate_method_results('persistence-model-regression_' + os.path.splitext(filename)[0], test_y,
                            persistence_model_result)

    arima_result = perform_arima_prediction(df, 'airly-pm1(t+0)', number_of_timestep_ahead)
    evaluate_method_results('_'.join(x_column_names) + '_arima_' + os.path.splitext(filename)[0], test_y, arima_result)

def compare_methods_each_iter(df, train_x, train_y, test_x, test_y, number_of_timestep_ahead, number_of_timestep_backward, filename, x_column_names):
    persistence_model_result = perform_persistence_model_prediction(df, 'airly-pm1(t-1)', len(test_y),
                                                                    number_of_timestep_ahead)
    evaluate_method_results('persistence-model-regression_' + os.path.splitext(filename)[0], test_y,
                            persistence_model_result)

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

    #print(test_y)
    #print(persistence_model_result)
    #print(linear_regression_result)
    #print(random_forest_regression_result)
    #print(nn_lstm_regression_result)
    #print(nn_mlp_regression_result)

if __name__ == "__main__":
    directory = './data/multivariate-time-series'

    x = []
    y = []

    for filename in os.listdir(directory):
        print(filename)
        if filename == 'pollution.csv' or filename == 'raw-626.csv' or filename == 'raw-210.csv' or filename == 'raw-218.csv' or filename == 'raw-559.csv':
            continue
        csv = os.path.join(directory, filename)
        df = read_csv(csv, header=0, index_col=0)
        df.index = to_datetime(df.index)

        df = df.apply(lambda ts: ts.interpolate(method='nearest'))
        df = df.apply(lambda ts: ts.resample('1H').nearest())
       # if 'ow-wnd-spd' in df.columns and 'ow-wnd-deg' in df.columns:
       #     df['ow-wnd-x'] = df.apply(lambda row: row['ow-wnd-spd'] * math.cos(math.radians(row['ow-wnd-deg'])), axis=1)
       #     df['ow-wnd-y'] = df.apply(lambda row: row['ow-wnd-spd'] * math.sin(math.radians(row['ow-wnd-deg'])), axis=1)

        y_column_names = ['airly-pm1']
        x_column_names = ['airly-pm1','ow-wnd-spd', 'here-traffic-jam', 'airly-tmp']

        number_of_timestep_ahead = 24
        number_of_timestep_backward = 24

        x_length = len(x_column_names) * number_of_timestep_backward
        y_length = len(y_column_names) * number_of_timestep_ahead

        for i in range(int(0.2 *df.values.shape[0]),0,-1):

            partial_df = df.iloc[-(i+1)-(30*24):-(i+1)][:]

            new_df = transform_dataframe_to_supervised(partial_df, x_column_names, y_column_names, number_of_timestep_ahead,
                                                number_of_timestep_backward)

            new_df = new_df.dropna()

            number_of_rows = new_df.values.shape[0]
            number_of_train_rows = number_of_rows - 1

            train_x = new_df.values[:number_of_train_rows - number_of_timestep_ahead, :x_length]
            train_y = new_df.values[:number_of_train_rows - number_of_timestep_ahead, -y_length:]

            test_x = new_df.values[number_of_train_rows:, :x_length]
            test_y = new_df.values[number_of_train_rows:, -y_length:]

            #print(test_x, test_y)

            import numpy as np

            #new_df.to_csv("base_" + str(i) + '_' + filename + '.csv')
            #np.savetxt("base_1_" + str(i) + '_' + filename + '.csv', new_df.values, delimiter=",")
            #np.savetxt("train_x_" + str(i) + '_' + filename + '.csv', train_x, delimiter=",")
            #np.savetxt("train_y_" + str(i) + '_' + filename + '.csv', train_y, delimiter=",")
            #np.savetxt("test_x_" + str(i) + '_' + filename + '.csv', test_x, delimiter=",")
            #np.savetxt("test_y_" + str(i) + '_' + filename + '.csv', test_y, delimiter=",")

            compare_methods_each_iter(new_df, train_x, train_y, test_x, test_y, number_of_timestep_ahead, number_of_timestep_backward, str(i) + '_' + filename, x_column_names)


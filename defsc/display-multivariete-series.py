import os

import itertools
from pandas import read_csv, to_datetime

from defsc.data_structures_transformation.data_structures_transformation import transform_dataframe_to_supervised, \
    split_timeseries_set_on_test_train
from defsc.filtering.fill_missing_values import simple_fill_missing_values
from defsc.time_series_forecasting.forecasts import perform_persistence_model_prediction, evaluate_method_results, \
    perform_arima_prediction, perform_linear_regression_prediction, perform_random_forest_regression_prediction, \
    perform_nn_lstm_prediction, perform_nn_mlp_prediction
from defsc.visualizations.time_series_visualization import plot_all_time_series_from_dataframe, \
    plot_all_time_series_decomposition_from_dataframe
import statsmodels.api as sm
from matplotlib import pyplot

def compare_methods_once(df, train_x, train_y, test_x, test_y, number_of_timestep_ahead, number_of_timestep_backward, filename, x_column_names):
    persistence_model_result = perform_persistence_model_prediction(df, 'airly-pm1(t-1)', len(test_y),
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
        print(filename)

        if filename == 'pollution.csv' or filename == 'raw-626.csv' or filename == 'raw-210.csv':
            continue
        csv = os.path.join(directory, filename)
        df = read_csv(csv, header=0, index_col=0)
        df.index = to_datetime(df.index)

        df = df.apply(lambda ts: ts.interpolate(method='nearest'))
        df = df.apply(lambda ts: ts.resample('1H').nearest())

        #df['here-traffic-speed'] = df['here-traffic-speed'] * 100
        #df['airly-pm1'] = df['airly-pm1'] * 1.8


        if 'ow-vis' in df.columns and 'here-traffic-speed' in df.columns:
            df['airly-pm10'].plot()
            pyplot.draw()
            df['airly-pm1'].plot()
            pyplot.draw()

        pyplot.show()


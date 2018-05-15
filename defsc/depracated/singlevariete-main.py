import os

import itertools
from pandas import read_csv, to_datetime
import statsmodels.api as sm

from defsc.data_structures_transformation.data_structures_transformation import transform_dataframe_to_supervised, \
    split_timeseries_set_on_test_train
from defsc.filtering.time_series_cleaning import simple_fill_missing_values
from defsc.time_series_forecasting.forecasts import perform_persistence_model_prediction, evaluate_method_results, \
    perform_arima_prediction, perform_linear_regression_prediction, perform_random_forest_regression_prediction, \
    perform_nn_lstm_prediction, perform_nn_mlp_prediction, compare_methods
from defsc.visualizations.time_series_visualization import plot_all_time_series_from_dataframe, \
    plot_all_time_series_decomposition_from_dataframe
import matplotlib.pyplot as pyplot
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import (naive,
                                         drift,
                                         mean,
                                         seasonal_naive)

if __name__ == "__main__":
    directory = './data/singlevariate-time-series'
    for filename in sorted(os.listdir(directory)):
        csv = os.path.join(directory, filename)
        df = read_csv(csv, header=0, index_col=0)
        df.index = to_datetime(df.index)

        df = df.apply(lambda ts: ts.interpolate(method='nearest'))
        df = df.apply(lambda ts: ts.resample('1H').nearest())
        df = df.dropna()

        #plot_all_time_series_from_dataframe(df)
        #plot_all_time_series_decomposition_from_dataframe(df)

        number_of_timestep_ahead = 24
        number_of_timestep_backward = 24

        x_column_names = df.columns
        y_column_names = df.columns

        df = transform_dataframe_to_supervised(df, x_column_names, y_column_names, number_of_timestep_ahead,
                                               number_of_timestep_backward)

        df = df.dropna()

        train_x, train_y, test_x, test_y = split_timeseries_set_on_test_train(df.values,
                                                                              len(
                                                                                  x_column_names) * number_of_timestep_backward,
                                                                              len(
                                                                                  y_column_names) * number_of_timestep_ahead,
                                                                              number_of_timestep_ahead,
                                                                              percantage_of_train_data=0.95)

        compare_methods(df, train_x, train_y, test_x, test_y, number_of_timestep_ahead, number_of_timestep_backward, filename, x_column_names, y_column_names[0])

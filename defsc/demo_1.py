import os

import itertools

from fancyimpute import KNN
from pandas import read_csv, to_datetime, DataFrame, date_range, DatetimeIndex
import numpy as np

from defsc.data_structures_transformation.data_structures_transformation import transform_dataframe_to_supervised, \
    split_timeseries_set_on_test_train
from defsc.filtering.time_series_cleaning import simple_fill_missing_values, drop_missing_values, \
    fill_missing_values_using_mean_before_after
from defsc.time_series_forecasting.forecasts import perform_persistence_model_prediction, evaluate_method_results, \
    perform_arima_prediction, perform_linear_regression_prediction, perform_random_forest_regression_prediction, \
    perform_nn_lstm_prediction, perform_nn_mlp_prediction
from defsc.visualizations.time_series_visualization import plot_all_time_series_from_dataframe, \
    plot_all_time_series_decomposition_from_dataframe
import statsmodels.api as sm
from matplotlib import pyplot

if __name__ == "__main__":
    directory = './data/multivariate-time-series-may'
    for filename in os.listdir(directory):
        print(filename)

        if (filename != 'raw-210.csv'):
            continue

        csv = os.path.join(directory, filename)
        df = read_csv(csv, header=0, index_col=0)
        df.index = to_datetime(df.index)

        df = drop_missing_values(df, df.columns,start='2017-09-23 00:00:00', end='2018-04-30 23:00:00', drop_column_factor=0.8)

        #df = df.reset_index()
        df.plot(subplots=True)
        pyplot.show()

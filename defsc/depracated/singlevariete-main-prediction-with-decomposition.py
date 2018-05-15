import os

import itertools
from pandas import read_csv, to_datetime
import statsmodels.api as sm

from defsc.data_structures_transformation.data_structures_transformation import transform_dataframe_to_supervised, \
    split_timeseries_set_on_test_train
from defsc.filtering.time_series_cleaning import simple_fill_missing_values
from defsc.time_series_forecasting.forecasts import perform_persistence_model_prediction, evaluate_method_results, \
    perform_arima_prediction, perform_linear_regression_prediction, perform_random_forest_regression_prediction, \
    perform_nn_lstm_prediction, perform_nn_mlp_prediction, compare_methods, perform_svr_regression_prediction
from defsc.visualizations.time_series_visualization import plot_all_time_series_from_dataframe, \
    plot_all_time_series_decomposition_from_dataframe
import matplotlib.pyplot as pyplot
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import (naive,
                                         drift,
                                         mean,
                                         seasonal_naive)

def perform_preditction(df):
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

    persistence_model_result = perform_persistence_model_prediction(df, y_column_names[0] + '(t-1)', len(test_y),
                                                                    number_of_timestep_ahead)

    linear_regression_result = perform_linear_regression_prediction(train_x, train_y, test_x,
                                                                    number_of_timestep_ahead)

    svr_regression_result = perform_svr_regression_prediction(train_x, train_y, test_x,
                                                                    number_of_timestep_ahead)

    random_forest_regression_result = perform_random_forest_regression_prediction(train_x, train_y, test_x,
                                                                                  number_of_timestep_ahead)
    nn_lstm_regression_result = perform_nn_lstm_prediction(train_x, train_y, test_x, test_y,
                                                           number_of_timestep_ahead, number_of_timestep_backward,
                                                           train_x.shape[1])

    nn_mlp_regression_result = perform_nn_mlp_prediction(train_x, train_y, test_x, test_y, number_of_timestep_ahead)

    return (persistence_model_result, linear_regression_result, svr_regression_result, random_forest_regression_result, nn_lstm_regression_result)


def get_seasonal_test_y(df):
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

    return test_y

if __name__ == "__main__":
    directory = './data/singlevariate-time-series'
    for filename in sorted(os.listdir(directory)):
        csv = os.path.join(directory, filename)
        df = read_csv(csv, header=0, index_col=0)
        df.index = to_datetime(df.index)

        df = df.apply(lambda ts: ts.interpolate(method='nearest'))
        df = df.apply(lambda ts: ts.resample('1H').nearest())
        df = df.dropna()


        number_of_timestep_ahead = 24
        number_of_timestep_backward = 24

        x_column_names = df.columns
        y_column_names = df.columns

        if (df.index.size >= 2 * 24 * 30 *12):
            freq = 24 * 30 * 12
        else:
            freq = 24

        stl_decompose = decompose(df, period=freq)

        stl_decompose.plot()
        pyplot.title(filename)
        pyplot.savefig('./results/' + filename + '.png')

        trend = stl_decompose.trend
        resid = stl_decompose.resid
        seasonal = stl_decompose.seasonal

        trend_result = perform_preditction(trend)
        resid_result = perform_preditction(resid)
        seasonal_result = get_seasonal_test_y(seasonal)
        real_result = get_seasonal_test_y(df)

        #print(trend_result[0].shape, seasonal_result.shape, resid_result[0].shape)
        if ((trend_result[0].shape == seasonal_result.shape and seasonal_result.shape == resid_result[0].shape)):
            aggregated_prediction_persistence_model = trend_result[0] + seasonal_result + resid_result[0]
            aggregated_prediction_linear_regression = trend_result[1] + seasonal_result + resid_result[1]
            aggregated_prediction_avr_regression = trend_result[2] + seasonal_result + resid_result[2]
            aggregated_prediction_random_forest_regression = trend_result[3] + seasonal_result + resid_result[3]
            aggregated_prediction_nn_lstm_regression = trend_result[4] + seasonal_result + resid_result[4]
            #aggregated_prediction_nn_mlp_regression = trend_result[5] + seasonal_result + resid_result[5]

            evaluate_method_results('_'.join(x_column_names) + '_aggregated_persistance' + os.path.splitext(filename)[0],
                                    real_result, aggregated_prediction_persistence_model)

            evaluate_method_results('_'.join(x_column_names) + '_aggregated_linear' + os.path.splitext(filename)[0],
                                    real_result, aggregated_prediction_linear_regression)

            evaluate_method_results('_'.join(x_column_names) + '_aggregated_avr' + os.path.splitext(filename)[0],
                                    real_result, aggregated_prediction_avr_regression)

            evaluate_method_results('_'.join(x_column_names) + '_aggregated_random_forest' + os.path.splitext(filename)[0],
                                    real_result, aggregated_prediction_random_forest_regression)

            evaluate_method_results('_'.join(x_column_names) + '_aggregated_nn_lstm' + os.path.splitext(filename)[0],
                                    real_result, aggregated_prediction_nn_lstm_regression)

            #evaluate_method_results('_'.join(x_column_names) + '_aggregated_nn_mlp' + os.path.splitext(filename)[0],
            #                        real_result, aggregated_prediction_nn_mlp_regression)
        else:
            print('Shape is nt correct')

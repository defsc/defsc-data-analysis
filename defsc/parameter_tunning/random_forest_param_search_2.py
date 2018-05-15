from datetime import time

import numpy as np
from pandas import to_datetime, read_csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from defsc.data_structures_transformation.data_structures_transformation import split_timeseries_set_on_test_train, \
    transform_dataframe_to_supervised
from defsc.filtering.time_series_cleaning import simple_fill_missing_values, drop_missing_values
from defsc.time_series_forecasting.forecasts import make_forecasts, evaluate_method_results


def multioutput_random_forest_regression_params_search(train_x, train_y):
    n_of_estimators = [10,50,100,200,300,400,500]
    n_of_features = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]

    for n_of_estimator in n_of_estimators:
        for n_of_feature in n_of_features:
            print(n_of_estimator, n_of_feature)
            regr = RandomForestRegressor(n_estimators=n_of_estimator, max_features=n_of_feature)
            model = MultiOutputRegressor(regr).fit(train_x, train_y)
            forecast = make_forecasts(model, test_x, number_of_timestep_ahead)
            evaluate_method_results('210' + '_radnom-forest-regression',
                                    test_y,
                                    forecast)





if __name__ == "__main__":
    csv = '../data/multivariate-time-series-may-wios/raw-airly-210.csv'
    df = read_csv(csv, header=0, index_col=0)
    df.index = to_datetime(df.index)

    number_of_timestep_ahead = 24
    number_of_timestep_backward = 24

    y_column_names = ['airly-pm10']
    x_history_column_names = ['airly-pm10']
    x_forecast_column_names = ['ow-wnd-spd', 'ow-tmp', 'ow-press', 'ow-hum']

    if not all(column in df.columns for column in (x_history_column_names + x_forecast_column_names)):
        print(csv)
        print('Not all columns')
        exit(1)

    df = drop_missing_values(df, x_history_column_names + x_forecast_column_names, start='2017-09-23 00:00:00',
                             end='2018-04-30 23:00:00')

    if not all(column in df.columns for column in (x_history_column_names + x_forecast_column_names)):
        print(csv)
        print('Not all columns after dropping missing values')
        exit(1)

    new_df = transform_dataframe_to_supervised(df, x_history_column_names, x_forecast_column_names, y_column_names,
                                               number_of_timestep_ahead,
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
    multioutput_random_forest_regression_params_search(train_x, train_y)

from numpy.linalg import LinAlgError
from pandas import read_csv, to_datetime

from defsc.data_structures_transformation.data_structures_transformation import transform_dataframe_to_supervised, \
    split_timeseries_set_on_test_train
from defsc.filtering.time_series_cleaning import simple_fill_missing_values
from defsc.time_series_forecasting.forecasts import perform_arima_prediction, evaluate_method_results

if __name__ == "__main__":
    csv = '../data/multivariate-time-series/raw-210.csv'
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

    for p in range(1,10,1):
        for d in range(3):
            for q in range(10):
                try:
                    arima_result = perform_arima_prediction(df, y_column_names[0] + '(t+0)', number_of_timestep_ahead,
                                                            p=p, d=d, q=q)
                    evaluate_method_results('p:{}_d:{}_q:{}'.format(p, d, q) + '_arima', test_y, arima_result)
                except (ValueError, LinAlgError):
                    print('Error in: p:{}_d:{}_q:{}'.format(p, d, q))

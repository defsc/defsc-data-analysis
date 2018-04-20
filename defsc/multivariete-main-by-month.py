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



if __name__ == "__main__":
    directory = './data/multivariate-time-series'

    x = []
    y = []

    for filename in os.listdir(directory):
        print(filename)
        if filename == 'pollution.csv':
            continue
        csv = os.path.join(directory, filename)
        df = read_csv(csv, header=0, index_col=0)
        df.index = to_datetime(df.index)

        df = df.apply(lambda ts: ts.interpolate(method='nearest'))
        df = df.apply(lambda ts: ts.resample('1H').nearest())
       # if 'ow-wnd-spd' in df.columns and 'ow-wnd-deg' in df.columns:
       #     df['ow-wnd-x'] = df.apply(lambda row: row['ow-wnd-spd'] * math.cos(math.radians(row['ow-wnd-deg'])), axis=1)
       #     df['ow-wnd-y'] = df.apply(lambda row: row['ow-wnd-spd'] * math.sin(math.radians(row['ow-wnd-deg'])), axis=1)

        months = {n: g
                  for n, g in df.groupby(TimeGrouper('M'))}

        for k in sorted(months.keys()):
            df_per_month = months[k]

            df_per_month = df_per_month.loc[:, df_per_month.apply(Series.nunique) != 1]
            df_per_month = df_per_month.dropna(axis=1, how='all')

#            x_parameter = 'airly-pm1'
#            y_parameter = 'here-traffic-jam'

#            if x_parameter in df_per_month.columns and y_parameter in df_per_month.columns:
#                scatter_plot_of_the_value_of_one_parameter_in_dependence_of_another(df_per_month, x_parameter, y_parameter)

            if 'airly-pm1' in df_per_month.columns and 'here-traffic-jam' in df_per_month.columns:

                y_column_names = ['airly-pm1']
                x_column_names = ['airly-pm1', 'ow-wnd-spd','here-traffic-jam','airly-tmp']

                number_of_timestep_ahead = 24
                number_of_timestep_backward = 24

                new_df = transform_dataframe_to_supervised(df_per_month, x_column_names, y_column_names, number_of_timestep_ahead,
                                                    number_of_timestep_backward)

                new_df = new_df.dropna()

                train_x, train_y, test_x, test_y = split_timeseries_set_on_test_train(new_df.values,
                                                                                   len(
                                                                                       x_column_names) * number_of_timestep_backward,
                                                                                   len(
                                                                                       y_column_names) * number_of_timestep_ahead,
                                                                                      0,
                                                                                      percantage_of_train_data=0.80)

                #import numpy
                #numpy.savetxt("array.csv", test_y, delimiter=",")
                #print(test_y)
                #import time
                #time.sleep(1000)

                compare_methods_each_iter(new_df, train_x, train_y, test_x, test_y, number_of_timestep_ahead, number_of_timestep_backward, str(k.month)+'_' + filename, x_column_names)

                #from matplotlib import pyplot
                #df_per_month[x_column_names[0]].plot()
                #pyplot.show()

            #if 'airly-tmp' in df_per_month.columns and 'airly-pm1' in df_per_month.columns:
            #    x.append(df_per_month['airly-tmp'].mean(skipna=True)
            #    y.append(df_per_month['airly-tmp'].corr(df_per_month['airly-pm1'],method='spearman'))


           # for column in df_per_month.columns:
           #     from matplotlib import pyplot

                #print(df_per_month[column].isna().sum())
                #print(df_per_month[column].notna().sum())
                #df_per_month[column].plot()
                #pyplot.title(filename + '_' + column + '_' + str(k.month))
                #pyplot.show()

            #print(df_per_month.mean(skipna=True))
            #title = filename + '_' + str(k.month)
            #plot_heat_map_of_correlation_coefficients(df_per_month, method='spearman', title=title)

            #for column in df_per_month.columns:
                #print('Null:',df_per_month[column].isna().sum())
                #print('Not Null:', df_per_month[column].notna   ().sum())
                #df_per_month[column].plot()
                #from matplotlib import pyplot
                #pyplot.title(column)
                #pyplot.show()


                #df = simple_fill_missing_values(df)

                #plot_all_time_series_from_dataframe(df)
                #plot_all_time_series_decomposition_from_dataframe(df)

                #number_of_timestep_ahead = 24
                #number_of_timestep_backward = 24

                #predicted_column = 'airly-pm1'

                # df[predicted_column] = sm.tsa.seasonal_decompose(df[predicted_column], model='additive', freq=24*30).trend
                #
                # columns = df.columns.tolist()
                # columns.remove(predicted_column)
                #
                # i = 0
                #
                # for combinations_len in range(len(columns)):
                #     for x_column_combination in itertools.combinations(columns, combinations_len  + 1):
                #         x_column_names = list(x_column_combination)
                #         x_column_names.append(predicted_column)
                #         y_column_names = [predicted_column]
                #
                #         new_df = transform_dataframe_to_supervised(df, x_column_names, y_column_names, number_of_timestep_ahead,
                #                                                number_of_timestep_backward)
                #
                #         new_df = new_df.dropna()
                #
                #         train_x, train_y, test_x, test_y = split_timeseries_set_on_test_train(new_df.values,
                #                                                                               len(
                #                                                                                   x_column_names) * number_of_timestep_backward,
                #                                                                               len(
                #                                                                                   y_column_names) * number_of_timestep_ahead)
                #         if i == 0:
                #             compare_methods_once(new_df, train_x, train_y, test_x, test_y, number_of_timestep_ahead,
                #                                       number_of_timestep_backward, filename, x_column_names)
                #
                #         compare_methods_each_iter(new_df, train_x, train_y, test_x, test_y, number_of_timestep_ahead, number_of_timestep_backward, filename, x_column_names)
                #

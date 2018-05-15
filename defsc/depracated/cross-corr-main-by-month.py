import math
import os

from pandas import read_csv, to_datetime, TimeGrouper, Series, scatter_matrix

from defsc.data_structures_transformation.data_structures_transformation import transform_dataframe_to_supervised, \
    split_timeseries_set_on_test_train
from defsc.visualizations.time_series_visualization import \
    scatter_plot_of_the_value_of_one_parameter_in_dependence_of_another, \
    plot_cross_corelation_between_all_parameters_accross_the_time, box_plot

if __name__ == "__main__":
    directory = './data/multivariate-time-series'

    for filename in os.listdir(directory):
        print(filename)

        if filename == 'pollution.csv' or filename == 'raw-210.csv' or filename == 'raw-218.csv':
            continue
        csv = os.path.join(directory, filename)
        df = read_csv(csv, header=0, index_col=0)
        df.index = to_datetime(df.index)

        df = df.apply(lambda ts: ts.interpolate(method='nearest'))
        df = df.apply(lambda ts: ts.resample('1H').nearest())

        if 'ow-wnd-spd' in df.columns and 'ow-wnd-deg' in df.columns:
            df['ow-wnd-x'] = df.apply(lambda row: row['ow-wnd-spd'] * math.cos(math.radians(row['ow-wnd-deg'])), axis=1)
            df['ow-wnd-y'] = df.apply(lambda row: row['ow-wnd-spd'] * math.sin(math.radians(row['ow-wnd-deg'])), axis=1)


        #from matplotlib import pyplot
        #df['airly-pm1'].plot()
        #df['here-traffic-jam'].apply(lambda x: x * 10).plot()
        #pyplot.show()

        months = {n: g
                  for n, g in df.groupby(TimeGrouper('M'))}

        #scatter_matrix(df, alpha=0.2,figsize=(26,26), diagonal='kde')
        #from matplotlib import pyplot
        #pyplot.show()

        if 'here-traffic-jam' in df.columns:
            box_plot(df, 'airly-pm10', 'here-traffic-jam', number_of_buckets=10)

        #scatter_plot_of_the_value_of_one_parameter_in_dependence_of_another(df, 'airly-tmp', 'airly-pm1')
        #bp = df.boxplot(column='airly-tmp', by='airly-pm1')
        #bp.plot()
        #from matplotlib import pyplot
        #print('Tuuu')
        #pyplot.show()


        # for k in sorted(months.keys()):
        #     print(k)
        #     df_per_month = months[k]
        #
        #     df_per_month = df_per_month.loc[:, df_per_month.apply(Series.nunique) != 1]
        #     df_per_month = df_per_month.dropna(axis=1, how='all')
        #
        #     df_per_month = df_per_month.apply(lambda ts: ts.truncate(after='2017-11-20 00:59:00+00:00'))
        #
        #     x_parameter = 'ow-vis'
        #     y_parameter = 'here-traffic-speed'
        #
        #     if x_parameter in df_per_month.columns:
        #         box_plot(df_per_month, y_parameter, x_parameter, number_of_buckets=5)
        #         df_per_month[x_parameter].plot()
        #         df_per_month[y_parameter] = df_per_month[y_parameter] * 100
        #         df_per_month[y_parameter].plot()
        #         from matplotlib import pyplot
        #
        #         pyplot.show()
        #
        #         plot_cross_corelation_between_all_parameters_accross_the_time(df_per_month, x_parameter, 50, method='spearman')

            #if x_parameter in df_per_month.columns and y_parameter in df_per_month.columns:
            #    scatter_plot_of_the_value_of_one_parameter_in_dependence_of_another(df_per_month, x_parameter, y_parameter)


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

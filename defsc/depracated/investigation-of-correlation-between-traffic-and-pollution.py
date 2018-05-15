from builtins import bytearray

import dateutil
from pandas import read_csv, to_datetime, Series, TimeGrouper, scatter_matrix, cut
from pymongo import MongoClient
from stldecompose import decompose

from defsc.visualizations.time_series_visualization import \
    plot_cross_corelation_between_all_parameters_accross_the_time, \
    scatter_plot_of_the_value_of_one_parameter_in_dependence_of_another, box_plot


def load_df_with_pm():
    filename = './defsc/data/singlevariate-time-series/MpKrakAlKras-PM10.csv'

    df = read_csv(filename, header=0, index_col=0)
    df.index = to_datetime(df.index)

    df = df.apply(lambda ts: ts.interpolate(method='nearest'))
    df = df.apply(lambda ts: ts.resample('1H').nearest())
    df = df.dropna()

    return df

def load_df_with_traffic_jam():
    client = MongoClient('localhost', 27017)
    db = client['local']
    traffic_measurements = db['traffic_measurements']

    time_series = {}

    #'lom_id': "1124"
    for measurement in traffic_measurements.find({}):
        #timestamp = dateutil.parser.parse(measurement['CREATED_TIMESTAMP'])
        timestamp = measurement['_id'].generation_time

        for roadway in measurement['RWS'][0]['RW']:
            if roadway['LI'] == '305+33561':
                for flow_item in roadway['FIS'][0]['FI']:
                    if flow_item['TMC']['PC'] == 32294:
                        #print(flow_item['TMC']['DE'])
                        #print(flow_item['CF'][0]['SP'])
                        time_series[timestamp] = flow_item['CF'][0]['JF']

    ts = Series(time_series)
    ts = ts.interpolate(method='nearest')
    ts = ts.resample('1H').nearest()
    ts = ts.tz_localize(None)

    return ts

if __name__ == "__main__":
    traffic_ts = load_df_with_traffic_jam()
    pm_df = load_df_with_pm()

    print(traffic_ts.index[0])
    print(traffic_ts.index[-1])

    print(pm_df.index[0])
    print(pm_df.index[-1])

    start_timestamp = '2017-09-23 12:00:00'
    end_timestamp = '2018-03-31 00:00:00'

    pm_df = pm_df.apply(lambda ts: ts.truncate(before=start_timestamp))
    pm_df = pm_df.apply(lambda ts: ts.truncate(after=end_timestamp))

    traffic_ts = traffic_ts.truncate(before=start_timestamp)
    traffic_ts = traffic_ts.truncate(after=end_timestamp)

    #decomposed_traffic_ts = decompose(traffic_ts, period=24)
    #decomposed_traffic_ts.plot()
    #from matplotlib import pyplot
    #pyplot.show()

    #traffic_ts = decomposed_traffic_ts.trend + decomposed_traffic_ts.resid
    pm_df['traffic'] = traffic_ts

    box_plot(pm_df, 'MpKrakAlKras-PM10', 'traffic', number_of_buckets=7)

    scatter_matrix(pm_df, figsize=(26, 26))
    from matplotlib import pyplot
    pyplot.show()

    plot_cross_corelation_between_all_parameters_accross_the_time(pm_df, 'MpKrakAlKras-PM10', 200,
                                                                  method='pearson')

    #scatter_plot_of_the_value_of_one_parameter_in_dependence_of_another(pm_df, 'traffic', 'MpKrakAlKras-PM10')

    #print(pm_df.columns)

    months = {n: g
              for n, g in pm_df.groupby(TimeGrouper('M'))}

    for k in sorted(months.keys()):
        print(k.month)
        df_per_month = months[k]

        df_per_month.plot()
        from matplotlib import pyplot
        pyplot.show()

        #plot_cross_corelation_between_all_parameters_accross_the_time(df_per_month, 'MpKrakAlKras-PM10', 200,
        #                                                              method='pearson')
        #scatter_plot_of_the_value_of_one_parameter_in_dependence_of_another(df_per_month,'traffic','MpKrakAlKras-PM10')

        box_plot(df_per_month, 'MpKrakAlKras-PM10', 'traffic', number_of_buckets=7)

    print('End')

import os
import json

import dateutil
import ephem
import math
from matplotlib import pyplot
from pandas import read_csv, to_datetime, Series, scatter_matrix

from defsc.visualizations.time_series_visualization import plot_heat_map_of_correlation_coefficients, \
    plot_cross_corelation_between_all_parameters_accross_the_time

def day_or_night(timestamp):
    sun = ephem.Sun()
    observer = ephem.Observer()
    # ↓ Define your coordinates here ↓
    observer.lat, observer.lon, observer.elevation = '50.049683', '19.944544', 209
    # ↓ Set the time (UTC) here ↓
    observer.date = timestamp.tz_localize('Europe/Warsaw').tz_convert('utc')
    sun.compute(observer)
    current_sun_alt = sun.alt
    magic_coef = current_sun_alt*180/math.pi

    if magic_coef < -6:
        # night
        return 0

    # day
    return 50

def load_wios_measurements_into_df(wios_csv_path):
    wios_df = read_csv(wios_csv_path, header=0, index_col=0)
    wios_df.index = to_datetime(wios_df.index)

    return wios_df

def extract_average_from_json_with_kras_station_traffic_data(measurement_list):
    time_series = {}

    for measurement in measurement_list:
        timestamp = dateutil.parser.parse(measurement['CREATED_TIMESTAMP'])

        for roadway in measurement['RWS'][0]['RW']:
            if roadway['LI'] == '305+33561':
                for flow_item in roadway['FIS'][0]['FI']:
                    if flow_item['TMC']['PC'] == 32294:
                        #print(flow_item['TMC']['DE'])
                        #print(flow_item['CF'][0]['SP'])
                        time_series[timestamp] = flow_item['CF'][0]['JF']

    ts = Series(time_series)
    ts = ts.tz_localize(None)
    ts = ts.resample('1H').mean()


    return ts

def load_df_with_traffic_jam(traffic_dir_path):
    measurements_kras_station = []
    measurements_bujaka_station = []

    for filename in os.listdir(traffic_dir_path):
        with open(os.path.join(traffic_dir_path, filename)) as f:
            measurement = data = json.load(f)
            if 'Kras' in filename:
                measurements_kras_station.append(measurement)
            if 'Bujaka' in filename:
                measurements_bujaka_station.append(measurement)

    kras_ts = extract_average_from_json_with_kras_station_traffic_data(measurements_kras_station)
    #bujaka_ts = extract_average_from_json_with_bujaka_station_traffic_data(measurements_bujaka_station)

    #return kras_ts, bujaka_ts

    return kras_ts



if __name__ == "__main__":
    msc_data_dir = os.environ['MSC_DATA']
    traffic_dir_path = os.path.join(msc_data_dir, 'air-pollution-traffic', 'traffic')
    wios_csv_path = os.path.join(msc_data_dir, 'air-pollution-traffic', 'air-pollution', 'wios-measurements.csv')

    wios_df = load_wios_measurements_into_df(wios_csv_path)
    kras_ts = load_df_with_traffic_jam(traffic_dir_path)
    wios_df['traffic'] = kras_ts
    wios_df = wios_df.dropna()
    wios_df['day_or_night'] = wios_df.index.map(day_or_night)


    #tmp_wios_df = wios_df.reset_index()

    wios_df.plot()
    pyplot.show()

    plot_heat_map_of_correlation_coefficients(wios_df)

    scatter_matrix(wios_df)
    pyplot.show()

    print(wios_df.columns)

    df = wios_df[['traffic','Kraków, Aleja Krasińskiego - pył zawieszony PM10','day_or_night']]
    df['traffic'] = df['traffic'] * 10
    df.plot()
    pyplot.show()

    plot_cross_corelation_between_all_parameters_accross_the_time(df, 'traffic', lag=200, method='pearson')

    wios_csv_path_april = './data/air-pollution-traffic/air-pollution/wios-measurements-april.csv'
    wios_april_df = load_wios_measurements_into_df(wios_csv_path_april)
    wios_april_df = wios_april_df.dropna()
    wios_april_df['day_or_night'] = wios_april_df.index.map(day_or_night)
    wios_april_df = wios_april_df[['Kraków, Aleja Krasińskiego - pył zawieszony PM10', 'day_or_night']]
    wios_april_df.plot()
    pyplot.show()

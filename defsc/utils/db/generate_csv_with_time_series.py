import dateutil.parser
import gpxpy.geo
import statsmodels.api as sm
import numpy as np
import pytz
import seaborn as sns

from itertools import tee
from itertools import combinations
from matplotlib import pyplot
from pandas import Series, DataFrame, reset_option, set_option, DataFrame
from pymongo import MongoClient
from pylab import rcParams


def print_full(x):
    set_option('display.max_rows', len(x))
    print(x)
    reset_option('display.max_rows')


def time_series(extract_parameter_timestamp_function, extract_parameter_value_function, db_collection, lom_id):
    time_series = {}
    for measurement in db_collection.find({'lom_id': lom_id}):
        timestamp = extract_parameter_timestamp_function(measurement)
        if extract_parameter_value_function(measurement) != 'NA' and extract_parameter_value_function(
                measurement) != 'N/A':
            time_series[timestamp] = extract_parameter_value_function(measurement)
    return Series(time_series)


def traffic_speed_time_series(db_collection, lom_id):
    time_series = {}
    for measurement in db_collection.find({'lom_id': lom_id}):
        timestamp = dateutil.parser.parse(measurement['CREATED_TIMESTAMP'])
        speed_sum = 0.0
        speed_weight = 0.0
        for roadway in measurement['RWS'][0]['RW']:
            for flow_item in roadway['FIS'][0]['FI']:
                speed_weight += flow_item['TMC']['LE']
                speed_sum += flow_item['TMC']['LE'] * flow_item['CF'][0].get('SU', 1.0)
        time_series[timestamp] = speed_sum / speed_weight
    return Series(time_series)


def traffic_jam_time_series(db_collection, lom_id):
    time_series = {}
    for measurement in db_collection.find({'lom_id': lom_id}):
        timestamp = dateutil.parser.parse(measurement['CREATED_TIMESTAMP'])
        jam_sum = 0.0
        jam_weight = 0.0
        for roadway in measurement['RWS'][0]['RW']:
            for flow_item in roadway['FIS'][0]['FI']:
                jam_weight += flow_item['TMC']['LE']
                jam_sum += flow_item['TMC']['LE'] * flow_item['CF'][0]['JF']
        time_series[timestamp] = jam_sum / jam_weight
    return Series(time_series)


def find_nearest_wunderground_sensor_for_airly_sensor(airly_db_collection, wunderground_db_collection):
    mapping = {}
    for airly_sensor in airly_db_collection.find():
        min_dist = float('Inf')
        for wunder_sensor in wunderground_db_collection:
            dist = gpxpy.geo.haversine_distance(wunder_sensor['location']['latitude'],
                                                wunder_sensor['location']['longitude'],
                                                airly_sensor['location']['latitude'],
                                                airly_sensor['location']['longitude'])
            if dist < min_dist:
                min_dist = dist
                mapping[str(airly_sensor['_id'])] = wunder_sensor['_id']
    return mapping


def find_nearest_wios_sensor_for_airly_sensor(airly_db_collection, wios_db_collection):
    mapping = {}
    for airly_sensor in airly_db_collection.find():
        min_dist = float('Inf')
        for wios_sensor in wios_db_collection.find():
            dist = gpxpy.geo.haversine_distance(airly_sensor['location']['latitude'],
                                                airly_sensor['location']['longitude'],
                                                float(wios_sensor['location']['latitude']),
                                                float(wios_sensor['location']['longitude']))
            if dist < min_dist:
                min_dist = dist
                mapping[airly_sensor['_id']] = wios_sensor['_id']
    return mapping


def extract_timestamp_from_airly_measurement(measurement):
    return dateutil.parser.parse(measurement['tillDateTime'])


def extract_timestamp_from_wios_measurement(measurement):
    return pytz.utc.localize(dateutil.parser.parse(measurement['timestamp']))


def extract_timestamp_from_mongo_id(measurement):
    return measurement['_id'].generation_time


def extract_pm1_from_airly_measurement(measurement):
    return measurement.get('measurements', {}).get('pm1', 'NA')


def extract_pm10_from_wios_measurement(measurement):
    if measurement.get('values', {}).get('PM10') == '':
        return 0.0
    return measurement.get('values', {}).get('PM10', 'NA')


def extract_pm10_from_airly_measurement(measurement):
    return measurement.get('measurements', {}).get('pm10', 'NA')


def extract_pm25_from_airly_measurement(measurement):
    return measurement.get('measurements', {}).get('pm25', 'NA')


def extract_pressure_from_airly_measurement(measurement):
    return measurement.get('measurements', {}).get('pressure', 'NA')


def extract_temperature_from_airly_measurement(measurement):
    return measurement.get('measurements', {}).get('temperature', 'NA')


def extract_humidity_from_airly_measurement(measurement):
    return measurement.get('measurements', {}).get('humidity', 'NA')


def extract_temp_from_ow_measurement(measurement):
    return measurement.get('main', {}).get('temp')


def extract_pressure_from_ow_measurement(measurement):
    return measurement.get('main', {}).get('pressure')


def extract_humidity_from_ow_measurement(measurement):
    return measurement.get('main', {}).get('humidity')


def extract_visibility_from_ow_measurement(measurement):
    return measurement.get('visibility')


def extract_wind_speed_from_ow_measurement(measurement):
    return measurement.get('wind', {}).get('speed')


def extract_wind_deg_from_ow_measurement(measurement):
    return measurement.get('wind', {}).get('deg')


def extract_windchill_from_wunder_measurement(measurement):
    return measurement.get('current_observation', {}).get('windchill_c', 'NA')


def extract_wind_kph_from_wunder_measurement(measurement):
    return measurement.get('current_observation', {}).get('wind_kph')


def extract_wind_dir_from_wunder_measurement(measurement):
    return measurement.get('current_observation', {}).get('wind_dir')


def extract_visibility_km_from_wunder_measurement(measurement):
    if measurement.get('current_observation', {}).get('visibility_km') != 'N/A' and measurement.get(
            'current_observation', {}).get('visibility_km') != '':
        return measurement.get('current_observation', {}).get('visibility_km')


def extract_uv_from_wunder_measurement(measurement):
    return measurement.get('current_observation', {}).get('UV', 'NA').replace('--','')


def extract_temp_c_from_wunder_measurement(measurement):
    return measurement.get('current_observation', {}).get('temp_c')


def extract_feelslike_c_from_wunder_measurement(measurement):
    if (measurement.get('current_observation', {}).get('feelslike_c') != ''):
        return measurement.get('current_observation', {}).get('feelslike_c')


def extract_relative_humidity_from_wunder_measurement(measurement):
    return measurement.get('current_observation', {}).get('relative_humidity', "0%").replace('%', '')


def extract_pressure_mb_from_wunder_measurement(measurement):
    return measurement.get('current_observation', {}).get('pressure_mb')


def extract_wind_gust_kph_from_wunder_measurement(measurement):
    return measurement.get('current_observation', {}).get('wind_gust_kph')


def extract_precip_1hr_in_from_wunder_measurement(measurement):
    if (measurement.get('current_observation', {}).get('precip_1hr_in') != ''):
        return measurement.get('current_observation', {}).get('precip_1hr_in')


def extract_precip_today_in_from_wunder_measurement(measurement):
    if (measurement.get('current_observation', {}).get('precip_today_in') != ''):
        return measurement.get('current_observation', {}).get('precip_today_in')


def process(ts):
    # ts = ts.truncate(before='2017-12-07 10:25:19+00:00')
    # ts = ts.truncate(after='2018-01-30 23:00:00+00:00')
    # ts = ts.resample('1H').mean()
    # ts = ts.fillna(ts.bfill())
    # decomposition = sm.tsa.seasonal_decompose(ts, model='additive')
    # fig = decomposition.plot()
    # pyplot.show()
    return ts


def save_dict_into_csv(dict_of_time_series):
    for k, v in dict_of_time_series.items():
        process(v).to_csv(path=k + '.csv', index=False)


def save_dict_into_csv_2(dict_of_time_series, filename):
    new = {}
    for k, v in dict_of_time_series.items():
        new[k] = process(v)
    df = DataFrame(new)
    df.to_csv('raw-' + filename + '.csv')


client = MongoClient('localhost', 27017)
db = client['local']

airly_sensors = db['airly_sensors']
wunder_sensors = list(db['wunder_sensors_1'].find()) + list(db['wunder_sensors_2'].find())
wios_sensors = db['wios_sensors']
airly_measurements = db['airly_sensors_measurements']
wios_measurements = db['wios_measurements']
ow_weather_measurements = db['open_weather_measurements']
wunder_weather_measurements = db['wunder_sensors_measurements']
traffic_measurements = db['traffic_measurements']

nearest_wunder_sensors = find_nearest_wunderground_sensor_for_airly_sensor(airly_sensors, wunder_sensors)
nearest_wios_sensor = find_nearest_wios_sensor_for_airly_sensor(airly_sensors, wios_sensors)

for airly_sensor in airly_sensors.find():
    time_series_dict = {}
    lom_id = airly_sensor['_id']
    nearest_wios_sensor_id = nearest_wios_sensor[lom_id]
    lom_id = str(lom_id)
    print(lom_id)

    dictionaries_with_measurements = {}
    for wios_measurement in wios_measurements.find({'station_id': str(nearest_wios_sensor_id)}):
        time_stamp = extract_timestamp_from_wios_measurement(wios_measurement)

        k = wios_measurement['measurement']['type']
        v = wios_measurement['measurement']['value']

        tmp_dict = dictionaries_with_measurements.get('wios-' + k, {})
        tmp_dict[time_stamp] = v
        dictionaries_with_measurements['wios-' + k] = tmp_dict

    for k, v in dictionaries_with_measurements.items():
        series = Series(v)
        series = series.astype(float)
        if not series.empty:
            time_series_dict[k] = series

    series = Series(
        time_series(extract_timestamp_from_airly_measurement, extract_pm1_from_airly_measurement, airly_measurements,
                    lom_id))
    if not series.empty:
        time_series_dict['airly-pm1'] = series

    series = Series(
        time_series(extract_timestamp_from_airly_measurement, extract_pm10_from_airly_measurement, airly_measurements,
                    lom_id))
    if not series.empty:
        time_series_dict['airly-pm10'] = series

    series = Series(
        time_series(extract_timestamp_from_airly_measurement, extract_pm25_from_airly_measurement, airly_measurements,
                    lom_id))
    if not series.empty:
        time_series_dict['airly-pm25'] = series

    series = Series(time_series(extract_timestamp_from_airly_measurement, extract_temperature_from_airly_measurement,
                                airly_measurements, lom_id))
    if not series.empty:
        time_series_dict['airly-tmp'] = series

    series = Series(time_series(extract_timestamp_from_airly_measurement, extract_pressure_from_airly_measurement,
                                airly_measurements, lom_id))
    if not series.empty:
        time_series_dict['airly-press'] = series

    series = Series(time_series(extract_timestamp_from_airly_measurement, extract_humidity_from_airly_measurement,
                                airly_measurements, lom_id))
    if not series.empty:
        time_series_dict['airly-hum'] = series

    series = Series(traffic_jam_time_series(traffic_measurements, lom_id))
    if not series.empty:
        time_series_dict['here-traffic-jam'] = series

    series = Series(traffic_speed_time_series(traffic_measurements, lom_id))
    if not series.empty:
        time_series_dict['here-traffic-speed'] = series

    series = Series(
        time_series(extract_timestamp_from_mongo_id, extract_temp_from_ow_measurement, ow_weather_measurements, lom_id))
    if not series.empty:
        time_series_dict['ow-tmp'] = series

    series = Series(
        time_series(extract_timestamp_from_mongo_id, extract_pressure_from_ow_measurement, ow_weather_measurements,
                    lom_id))
    if not series.empty:
        time_series_dict['ow-press'] = series

    series = Series(
        time_series(extract_timestamp_from_mongo_id, extract_humidity_from_ow_measurement, ow_weather_measurements,
                    lom_id))
    if not series.empty:
        time_series_dict['ow-hum'] = series

    series = Series(
        time_series(extract_timestamp_from_mongo_id, extract_visibility_from_ow_measurement, ow_weather_measurements,
                    lom_id))
    if not series.empty:
        time_series_dict['ow-vis'] = series

    series = Series(
        time_series(extract_timestamp_from_mongo_id, extract_wind_speed_from_ow_measurement, ow_weather_measurements,
                    lom_id))
    if not series.empty:
        time_series_dict['ow-wnd-spd'] = series

    series = Series(
        time_series(extract_timestamp_from_mongo_id, extract_wind_deg_from_ow_measurement, ow_weather_measurements,
                    lom_id))
    if not series.empty:
        time_series_dict['ow-wnd-deg'] = series

    series = Series(time_series(extract_timestamp_from_mongo_id, extract_wind_kph_from_wunder_measurement,
                                wunder_weather_measurements, nearest_wunder_sensors[lom_id]))
    if not series.empty:
        time_series_dict['wg-wnd-spd'] = series

    series = Series(time_series(extract_timestamp_from_mongo_id, extract_temp_c_from_wunder_measurement,
                                wunder_weather_measurements, nearest_wunder_sensors[lom_id]))
    if not series.empty:
        time_series_dict['wg-tmp'] = series

    series = Series(time_series(extract_timestamp_from_mongo_id, extract_pressure_mb_from_wunder_measurement,
                                wunder_weather_measurements, nearest_wunder_sensors[lom_id]))
    if not series.empty:
        time_series_dict['wg-press'] = series

    series = Series(time_series(extract_timestamp_from_mongo_id, extract_wind_gust_kph_from_wunder_measurement,
                                wunder_weather_measurements, nearest_wunder_sensors[lom_id]))
    if not series.empty:
        time_series_dict['wg-wnd-gust'] = series

    series = Series(
        time_series(extract_timestamp_from_mongo_id, extract_uv_from_wunder_measurement, wunder_weather_measurements,
                    nearest_wunder_sensors[lom_id]))
    if not series.empty:
        time_series_dict['wg-uv'] = series

    series = Series(time_series(extract_timestamp_from_mongo_id, extract_visibility_km_from_wunder_measurement,
                                wunder_weather_measurements, nearest_wunder_sensors[lom_id]))
    if not series.empty:
        time_series_dict['wg-vis'] = series

    series = Series(time_series(extract_timestamp_from_mongo_id, extract_relative_humidity_from_wunder_measurement,
                                wunder_weather_measurements, nearest_wunder_sensors[lom_id]))
    if not series.empty:
        time_series_dict['wg-hum'] = series

    series = Series(time_series(extract_timestamp_from_mongo_id, extract_windchill_from_wunder_measurement,
                                wunder_weather_measurements, nearest_wunder_sensors[lom_id]))
    if not series.empty:
        time_series_dict['wg-windchill'] = series

    series = Series(time_series(extract_timestamp_from_mongo_id, extract_feelslike_c_from_wunder_measurement,
                                wunder_weather_measurements, nearest_wunder_sensors[lom_id]))
    if not series.empty:
        time_series_dict['wg-feelslike'] = series

    series = Series(time_series(extract_timestamp_from_mongo_id, extract_precip_1hr_in_from_wunder_measurement,
                                wunder_weather_measurements, nearest_wunder_sensors[lom_id]))
    if not series.empty:
        time_series_dict['wg-precip-1h'] = series

    series = Series(time_series(extract_timestamp_from_mongo_id, extract_precip_today_in_from_wunder_measurement,
                                wunder_weather_measurements, nearest_wunder_sensors[lom_id]))
    if not series.empty:
        time_series_dict['wg-precip-1day'] = series

    save_dict_into_csv_2(time_series_dict, lom_id)

import dateutil.parser
import gpxpy.geo
import statsmodels.api as sm

from matplotlib import pyplot
from pandas import Series
from pymongo import MongoClient
from pylab import rcParams

def display_airly_sensors(airly_sensors_db_collection):
    for airly_sensor in airly_sensors.find():
        print(airly_sensor['_id'],airly_sensor['address']['locality'],airly_sensor['address'].get('route'))

def time_series(extract_parameter_timestamp_function, extract_parameter_value_function, db_collection, lom_id):
    time_series = {}
    for measurement in db_collection.find({'lom_id':lom_id}):
        timestamp = extract_parameter_timestamp_function(measurement)
        if extract_parameter_value_function(measurement) != 'NA' and extract_parameter_value_function(measurement) != 'N/A':
            time_series[timestamp] = extract_parameter_value_function(measurement)
    return Series(time_series)

def traffic_speed_time_series(db_collection, lom_id):
    time_series = {}
    for measurement in db_collection.find({'lom_id':lom_id}):
        timestamp = dateutil.parser.parse(measurement['CREATED_TIMESTAMP'])
        speed_sum = 0.0
        speed_weight = 0.0
        for roadway in measurement['RWS'][0]['RW']:
            for flow_item in roadway['FIS'][0]['FI']:
                speed_weight += flow_item['TMC']['LE']
                speed_sum += flow_item['TMC']['LE'] * flow_item['CF'][0]['SU']
        time_series[timestamp] = speed_sum / speed_weight
    return Series(time_series)

def traffic_jam_time_series(db_collection, lom_id):
    time_series = {}
    for measurement in db_collection.find({'lom_id':lom_id}):
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

def extract_timestamp_from_airly_measurement(measurement):
    return dateutil.parser.parse(measurement['tillDateTime'])

def extract_timestamp_from_mongo_id(measurement):
    return measurement['_id'].generation_time

def extract_pm1_from_airly_measurement(measurement):
    return measurement.get('measurements',{}).get('pm1')

def extract_pm10_from_airly_measurement(measurement):
    return measurement.get('measurements',{}).get('pm10')

def extract_pm25_from_airly_measurement(measurement):
    return measurement.get('measurements',{}).get('pm25')

def extract_pressure_from_airly_measurement(measurement):
    return measurement.get('measurements',{}).get('pressure')

def extract_temperature_from_airly_measurement(measurement):
    return measurement.get('measurements',{}).get('temperature')

def extract_humidity_from_airly_measurement(measurement):
    return measurement.get('measurements',{}).get('humidity')

def extract_temp_from_ow_measurement(measurement):
    return measurement.get('main',{}).get('temp')

def extract_pressure_from_ow_measurement(measurement):
    return measurement.get('main',{}).get('pressure')

def extract_humidity_from_ow_measurement(measurement):
    return measurement.get('main',{}).get('humidity')

def extract_visibility_from_ow_measurement(measurement):
    return measurement.get('visibility')

def extract_wind_speed_from_ow_measurement(measurement):
    return measurement.get('wind',{}).get('speed')

def extract_wind_deg_from_ow_measurement(measurement):
    return measurement.get('wind',{}).get('deg')

def extract_windchill_from_wunder_measurement(measurement):
    if (measurement.get('current_observation',{}).get('windchill_c') != 'NA'):
        return measurement.get('current_observation',{}).get('windchill_c')

def extract_wind_kph_from_wunder_measurement(measurement):
    return measurement.get('current_observation',{}).get('wind_kph')

def extract_wind_dir_from_wunder_measurement(measurement):
    return measurement.get('current_observation',{}).get('wind_dir')

def extract_visibility_km_from_wunder_measurement(measurement):
    return measurement.get('current_observation',{}).get('visibility_km')

def extract_uv_from_wunder_measurement(measurement):
    return measurement.get('current_observation',{}).get('UV')

def extract_temp_c_from_wunder_measurement(measurement):
    return measurement.get('current_observation',{}).get('temp_c')

def extract_feelslike_c_from_wunder_measurement(measurement):
    if (measurement.get('current_observation',{}).get('feelslike_c') != ''):
        return measurement.get('current_observation',{}).get('feelslike_c')

def extract_relative_humidity_from_wunder_measurement(measurement):
    return measurement.get('current_observation',{}).get('relative_humidity',"0%").replace('%', '')

def extract_pressure_mb_from_wunder_measurement(measurement):
    return measurement.get('current_observation',{}).get('pressure_mb')

def extract_wind_gust_kph_from_wunder_measurement(measurement):
    return measurement.get('current_observation',{}).get('wind_gust_kph')

def extract_precip_1hr_in_from_wunder_measurement(measurement):
    if (measurement.get('current_observation',{}).get('precip_1hr_in') != ''):
        return measurement.get('current_observation',{}).get('precip_1hr_in')

def extract_precip_today_in_from_wunder_measurement(measurement):
    if (measurement.get('current_observation',{}).get('precip_today_in') != ''):
        return measurement.get('current_observation',{}).get('precip_today_in')


client = MongoClient('localhost', 27017)
db = client['local']

airly_sensors = db['airly_sensors']
wunder_sensors = list(db['wunder_sensors_1'].find()) + list(db['wunder_sensors_2'].find())

airly_measurements = db['airly_sensors_measurements']
ow_weather_measurements = db['open_weather_measurements']
wunder_weather_measurements = db['wunder_sensors_measurements']
traffic_measurements = db['traffic_measurements']

nearest_wunder_sensors = find_nearest_wunderground_sensor_for_airly_sensor(airly_sensors, wunder_sensors)
lom_id = '195'
rcParams['figure.figsize'] = 11, 9

series = Series(time_series(extract_timestamp_from_airly_measurement, extract_pm1_from_airly_measurement, airly_measurements, lom_id))
series.plot()
pyplot.title('Airly - PM1')
pyplot.show()

series = series.fillna(series.bfill())
ax = series.plot()
#pyplot.title('Airly - PM1 After Fill')
#pyplot.show()

series = Series(time_series(extract_timestamp_from_airly_measurement, extract_temperature_from_airly_measurement, airly_measurements, lom_id))
#series.plot()
#pyplot.title('Airly - Temperature')
#pyplot.show()

series = series.fillna(series.bfill())
series.plot(ax=ax)
pyplot.title('Airly - Temperature After Fill')
pyplot.show()

series = Series(time_series(extract_timestamp_from_airly_measurement, extract_pm10_from_airly_measurement, airly_measurements, lom_id))
series.plot()
pyplot.title('Airly - PM10')
pyplot.show()

series = series.fillna(series.bfill())
series.plot()
pyplot.title('Airly - PM10 After Fill')
pyplot.show()

series = Series(time_series(extract_timestamp_from_airly_measurement, extract_pm25_from_airly_measurement, airly_measurements, lom_id))
series.plot()
pyplot.title('Airly - PM25')
pyplot.show()

series = series.fillna(series.bfill())
series.plot()
pyplot.title('Airly - PM25 After Fill')
pyplot.show()

series = Series(time_series(extract_timestamp_from_airly_measurement, extract_pressure_from_airly_measurement, airly_measurements, lom_id))
series.plot()
pyplot.title('Airly - Pressure')
pyplot.show()

series = series.fillna(series.bfill())
series.plot()
pyplot.title('Airly - Pressure After Fill')
pyplot.show()

series = Series(time_series(extract_timestamp_from_airly_measurement, extract_humidity_from_airly_measurement, airly_measurements, lom_id))
series.plot()
pyplot.title('Airly - Humidity')
pyplot.show()

series = series.fillna(series.bfill())
series.plot()
pyplot.title('Airly - Humidity After Fill')
pyplot.show()

series = Series(traffic_jam_time_series(traffic_measurements, lom_id))
series.plot()
pyplot.title('HERE Traffic - Jam')
pyplot.show()

series = series.fillna(series.bfill())
series.plot()
pyplot.title('HERE Traffic - Jam After Fill')
pyplot.show()

series = Series(traffic_speed_time_series(traffic_measurements, lom_id))
series.plot()
pyplot.title('HERE Traffic - Speed')
pyplot.show()

series = series.fillna(series.bfill())
series.plot()
pyplot.title('HERE Traffic - Speed After Fill')
pyplot.show()

series = Series(time_series(extract_timestamp_from_mongo_id, extract_temp_from_ow_measurement, ow_weather_measurements, lom_id))
series.plot()
pyplot.title('OpenWeather - Temp')
pyplot.show()

series = series.fillna(series.bfill())
series.plot()
pyplot.title('OpenWeather - Temp After Fill')
pyplot.show()

series = Series(time_series(extract_timestamp_from_mongo_id, extract_pressure_from_ow_measurement, ow_weather_measurements, lom_id))
series.plot()
pyplot.title('OpenWeather - Pressure')
pyplot.show()

series = series.fillna(series.bfill())
series.plot()
pyplot.title('OpenWeather - Pressure After Fill')
pyplot.show()

series = Series(time_series(extract_timestamp_from_mongo_id, extract_humidity_from_ow_measurement, ow_weather_measurements, lom_id))
series.plot()
pyplot.title('OpenWeather - Humidity')
pyplot.show()

series = series.fillna(series.bfill())
series.plot()
pyplot.title('OpenWeather - Humidity After Fill')
pyplot.show()

series = Series(time_series(extract_timestamp_from_mongo_id, extract_visibility_from_ow_measurement, ow_weather_measurements, lom_id))
series.plot()
pyplot.title('OpenWeather - Visiblity')
pyplot.show()

series = series.fillna(series.bfill())
series.plot()
pyplot.title('OpenWeather - Visiblity After Fill')
pyplot.show()
             
series = Series(time_series(extract_timestamp_from_mongo_id, extract_wind_speed_from_ow_measurement, ow_weather_measurements, lom_id))
series.plot()
pyplot.title('OpenWeather - Wind Speed')
pyplot.show()

series = series.fillna(series.bfill())
series.plot()
pyplot.title('OpenWeather - Wind Speed After Fill')
pyplot.show()
             
series = Series(time_series(extract_timestamp_from_mongo_id, extract_wind_deg_from_ow_measurement, ow_weather_measurements, lom_id))
series.plot()
pyplot.title('OpenWeather - Wind Degree')
pyplot.show()

series = series.fillna(series.bfill())
series.plot()
pyplot.title('OpenWeather - Wind Degree After Fill')
pyplot.show()

series = Series(time_series(extract_timestamp_from_mongo_id, extract_wind_kph_from_wunder_measurement, wunder_weather_measurements, nearest_wunder_sensors[lom_id]))
series.plot()
pyplot.title('WundergroundWeather - Wind kph')
pyplot.show()

series = series.fillna(series.bfill())
series.plot()
pyplot.title('WundergroundWeather - Wind kph After Fill')
pyplot.show()

series = Series(time_series(extract_timestamp_from_mongo_id, extract_temp_c_from_wunder_measurement, wunder_weather_measurements, nearest_wunder_sensors[lom_id]))
series.plot()
pyplot.title('WundergroundWeather - Temp')
pyplot.show()

series = series.fillna(series.bfill())
series.plot()
pyplot.title('WundergroundWeather - Temp After Fill')
pyplot.show()

series = Series(time_series(extract_timestamp_from_mongo_id, extract_pressure_mb_from_wunder_measurement, wunder_weather_measurements, nearest_wunder_sensors[lom_id]))
series = series.astype(float)
series.plot()
pyplot.title('WundergroundWeather - Pressure')
pyplot.show()

series = series.fillna(series.bfill())
series.plot()
pyplot.title('WundergroundWeather - Pressure After Fill')
pyplot.show()

series = Series(time_series(extract_timestamp_from_mongo_id, extract_wind_gust_kph_from_wunder_measurement, wunder_weather_measurements, nearest_wunder_sensors[lom_id]))
series = series.astype(float)
series.plot()
pyplot.title('WundergroundWeather - Wind Gust kph')
pyplot.show()

series = series.fillna(series.bfill())
series.plot()
pyplot.title('WundergroundWeather - Wind Gust kph After Fill')
pyplot.show()

series = Series(time_series(extract_timestamp_from_mongo_id, extract_uv_from_wunder_measurement, wunder_weather_measurements, nearest_wunder_sensors[lom_id]))
series = series.astype(float)
series.plot()
pyplot.title('WundergroundWeather - UV')
pyplot.show()

series = series.fillna(series.bfill())
series.plot()
pyplot.title('WundergroundWeather - UV After Fill')
pyplot.show()

series = Series(time_series(extract_timestamp_from_mongo_id, extract_visibility_km_from_wunder_measurement, wunder_weather_measurements, nearest_wunder_sensors[lom_id]))
series = series.astype(float)
series.plot()
pyplot.title('WundergroundWeather - Visibility')
pyplot.show()

series = series.fillna(series.bfill())
series.plot()
pyplot.title('WundergroundWeather - Visibility After Fill')
pyplot.show()

series = Series(time_series(extract_timestamp_from_mongo_id, extract_relative_humidity_from_wunder_measurement, wunder_weather_measurements, nearest_wunder_sensors[lom_id]))
series = series.astype(float)
series.plot()
pyplot.title('WundergroundWeather - Humidity')
pyplot.show()

series = series.fillna(series.bfill())
series.plot()
pyplot.title('WundergroundWeather - Humidity After Fill')
pyplot.show()

series = Series(time_series(extract_timestamp_from_mongo_id, extract_windchill_from_wunder_measurement, wunder_weather_measurements, nearest_wunder_sensors[lom_id]))
series = series.astype(float)
series.plot()
pyplot.title('WundergroundWeather - Windchill')
pyplot.show()

series = series.fillna(series.bfill())
series.plot()
pyplot.title('WundergroundWeather - Windchill After Fill')
pyplot.show()

series = Series(time_series(extract_timestamp_from_mongo_id, extract_feelslike_c_from_wunder_measurement, wunder_weather_measurements, nearest_wunder_sensors[lom_id]))
series = series.astype(float)
series.plot()
pyplot.title('WundergroundWeather - Feelslike')
pyplot.show()

series = series.fillna(series.bfill())
series.plot()
pyplot.title('WundergroundWeather - Feelslike After Fill')
pyplot.show()

series = Series(time_series(extract_timestamp_from_mongo_id, extract_precip_1hr_in_from_wunder_measurement, wunder_weather_measurements, nearest_wunder_sensors[lom_id]))
series = series.astype(float)
series.plot()
pyplot.title('WundergroundWeather - Precip 1h Inches')
pyplot.show()

series = series.fillna(series.bfill())
series.plot()
pyplot.title('WundergroundWeather - Precip 1h Inches After Fill')
pyplot.show()

series = Series(time_series(extract_timestamp_from_mongo_id, extract_precip_today_in_from_wunder_measurement, wunder_weather_measurements, nearest_wunder_sensors[lom_id]))
series = series.astype(float)
series.plot()
pyplot.title('WundergroundWeather - Precip Today Inches')
pyplot.show()

series = series.fillna(series.bfill())
series.plot()
pyplot.title('WundergroundWeather - Precip Today Inches After Fill')
pyplot.show()

#series = Series(time_series(extract_timestamp_from_mongo_id, extract_wind_dir_from_wunder_measurement, wunder_weather_measurements, nearest_wunder_sensors[lom_id]))
#print(series)

#decomposition = sm.tsa.seasonal_decompose(series, model='additive')
#fig = decomposition.plot()
#pyplot.show()


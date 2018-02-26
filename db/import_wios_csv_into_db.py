import csv
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['local']
wios_measurements = db['new_wios_measurements']


def insert_measurement_from_sensor_bulwarowa_into_db(item, db_collection):
	#item = line.split(';')
	print(item)

	new_doc = {}
	new_doc['wios_sensor_id'] = '402'
	new_doc['timestamp'] = item[0]
	values = {}
	values['CO'] = item[1]
	values['NO2'] = item[2]
	values['NO'] = item[3]
	values['PM10_1'] = item[4]
	values['PM10_2'] = item[5]
	values['PM25'] = item[6]
	values['SO2'] = item[7]

	new_doc['values'] = values

	db_collection.insert_one(new_doc)

def insert_measurement_from_sensor_dietla_into_db(item, db_collection):
	#item = line.split(';')
	print(item)

	new_doc = {}
	new_doc['wios_sensor_id'] = '10121'
	new_doc['timestamp'] = item[0]
	values = {}
	values['NO2'] = item[1]
	values['PM10'] = item[2]
	values['NO'] = item[3]

	new_doc['values'] = values

	db_collection.insert_one(new_doc)

def insert_measurement_from_sensor_kaszow_into_db(item, db_collection):
	#item = line.split(';')
	print(item)

	new_doc = {}
	new_doc['wios_sensor_id'] = '10119'
	new_doc['timestamp'] = item[0]
	values = {}
	values['NO2'] = item[1]
	values['O'] = item[2]
	values['NO'] = item[3]

	new_doc['values'] = values

	db_collection.insert_one(new_doc)

def insert_measurement_from_sensor_krasinskiego_into_db(item, db_collection):
	#item = line.split(';')
	print(item)

	new_doc = {}
	new_doc['wios_sensor_id'] = '400'
	new_doc['timestamp'] = item[0]
	values = {}
	values['CO'] = item[1]
	values['NO2'] = item[2]
	values['NO'] = item[3]
	values['PM10'] = item[4]
	values['PM25'] = item[5]
	values['benzen'] = item[6]

	new_doc['values'] = values

	db_collection.insert_one(new_doc)

def insert_measurement_from_sensor_kurdwanow_into_db(item, db_collection):
	#item = line.split(';')
	print(item)

	new_doc = {}
	new_doc['wios_sensor_id'] = '401'
	new_doc['timestamp'] = item[0]
	values = {}
	values['O'] = item[1]
	values['PM10_1'] = item[2]
	values['NO'] = item[3]
	values['PM_10_2'] = item[4]
	values['PM25'] = item[5]
	values['SO2'] = item[5]

	new_doc['values'] = values

	db_collection.insert_one(new_doc)

def insert_measurement_from_sensor_piastow_into_db(item, db_collection):
	#item = line.split(';')
	print(item)

	new_doc = {}
	new_doc['wios_sensor_id'] = '10139'
	new_doc['timestamp'] = item[0]
	values = {}
	values['PM10_1'] = item[2]
	values['PM10_2'] = item[1]

	new_doc['values'] = values

	db_collection.insert_one(new_doc)

def insert_measurement_from_sensor_skawina_into_db(item, db_collection):
	#item = line.split(';')
	print(item)

	new_doc = {}
	new_doc['wios_sensor_id'] = '437'
	new_doc['timestamp'] = item[0]
	values = {}
	values['NO2'] = item[1]
	values['SO2'] = item[2]
	values['PM10'] = item[3]
	values['NO'] = item[4]

	new_doc['values'] = values

	db_collection.insert_one(new_doc)

def insert_measurement_from_sensor_telimeny_into_db(item, db_collection):
	#item = line.split(';')
	print(item)

	new_doc = {}
	new_doc['wios_sensor_id'] = '10435'
	new_doc['timestamp'] = item[0]
	values = {}
	values['PM10_1'] = item[1]
	values['PM10_2'] = item[2]

	new_doc['values'] = values

	db_collection.insert_one(new_doc)

def insert_measurement_from_sensor_zloty_rog_into_db(item, db_collection):
	#item = line.split(';')
	print(item)

	new_doc = {}
	new_doc['wios_sensor_id'] = '10123'
	new_doc['timestamp'] = item[0]
	values = {}
	values['PM10_1'] = item[2]
	values['PM10_2'] = item[1]

	new_doc['values'] = values

	db_collection.insert_one(new_doc)


map_file_to_function = {}
map_file_to_function['merged-bulwarowa-gios-pjp-data.csv'] = insert_measurement_from_sensor_kurdwanow_into_db
map_file_to_function['merged-dietla-gios-pjp-data.csv'] = insert_measurement_from_sensor_dietla_into_db
map_file_to_function['merged-kaszow-gios-pjp-data.csv'] = insert_measurement_from_sensor_kaszow_into_db
map_file_to_function['merged-krasinskiego-gios-pjp-data.csv'] = insert_measurement_from_sensor_krasinskiego_into_db
map_file_to_function['merged-kurdwanow-gios-pjp-data.csv'] = insert_measurement_from_sensor_kurdwanow_into_db
map_file_to_function['merged-piastow-gios-pjp-data.csv'] = insert_measurement_from_sensor_piastow_into_db
map_file_to_function['merged-skawina-gios-pjp-data.csv'] = insert_measurement_from_sensor_skawina_into_db
map_file_to_function['merged-telimeny-gios-pjp-data.csv'] = insert_measurement_from_sensor_telimeny_into_db
map_file_to_function['merged-zloty-rog-gios-pjp-data.csv'] = insert_measurement_from_sensor_zloty_rog_into_db

for k,v in map_file_to_function.items():
	print(k)
	with open(k, 'r') as f:
		reader = csv.reader(f)
		# skip header rows
		next(reader)
		next(reader)

		for row in reader:
			v(row, wios_measurements)
import os
from pandas import read_csv
from pandas import to_datetime
import matplotlib.pyplot as pyplot
import math
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['local']
wios_measurements = db['wios_measurements']

all_counter = 0
not_nan_counter = 0

for filename in os.listdir('./merged_old_2017_2018/'):
	filename_prefix = filename.replace('.csv','')
	
	splitted_filename_prefix = filename_prefix.split('-')
	station_id = splitted_filename_prefix[0]
	pollutant_type = splitted_filename_prefix[1]
	
	csv = os.path.join('./merged_old_2017_2018/', filename)
	df = read_csv(csv, header=0, index_col=0)
	df.index = to_datetime(df.index)

	for index, row in df.iterrows():
		all_counter += 1
		if (not math.isnan(row[0])):
			not_nan_counter += 1
			print(station_id, str(index), pollutant_type, row[0])

			new_doc = {}
			new_doc['station_id'] = station_id
			new_doc['timestamp'] = str(index)

			measurement = {}
			measurement['type'] = pollutant_type
			measurement['value'] = row[0]
			new_doc['measurement'] = measurement

			wios_measurements.insert_one(new_doc)

print(not_nan_counter, all_counter)

from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['local']
wios_measurements = db['wios_measurements']

station_ids = wios_measurements.distinct('station_id')
pollutant_types = wios_measurements.distinct('measurement.type')

print(station_ids)
print(pollutant_types)

check_sum = 0

for station_id in station_ids:
    for pollutant_type in pollutant_types:
        measurements = wios_measurements.find({'station_id':station_id, 'measurement.type':pollutant_type})
        if measurements.count() != 0:
            print(station_id, pollutant_type, measurements.count())
            check_sum += measurements.count()
print(check_sum)

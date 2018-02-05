import json
import requests

#mongoimport --db local --collection wios_sensors --type json --jsonArray --file wios_sensors.json

def configure_http_requests():
	s = requests.Session()
	a = requests.adapters.HTTPAdapter(max_retries=1)
	b = requests.adapters.HTTPAdapter(max_retries=1)
	s.mount('http://', a)
	s.mount('https://', b)

	return s

def execute_request(address, headers):
	session = configure_http_requests()

	r = session.get(address, headers=headers)

	#print(r.status_code)
	#print(r.headers)
	#print(r.content)

	return r.content

bbox_min_lat = 49.95
bbox_min_lon = 19.71
bbox_max_lat = 50.11
bbox_max_lon = 20.11

headers = {}

address = 'http://api.gios.gov.pl/pjp-api/rest/station/findAll'

response_json_string = execute_request(address, headers).decode('utf-8')

response_json = json.loads(response_json_string)

#print(response_json)
#print(len(response_json))

filtered_sensors = []
for sensor in response_json:
	if (float(sensor['gegrLat']) > bbox_min_lat and float(sensor['gegrLat']) < bbox_max_lat and float(sensor['gegrLon']) > bbox_min_lon and float(sensor['gegrLon']) < bbox_max_lon):
		wios_sensor = {}

		location = {}
		location['latitude'] = sensor['gegrLat']
		location['longitude'] = sensor['gegrLon']
		wios_sensor['location'] = location

		wios_sensor['_id'] = sensor['id']
		wios_sensor['stationName'] = sensor['stationName']

		address = {}
		address['route'] = sensor['addressStreet']
		address['locality'] = sensor['city']
		wios_sensor['address'] = address

		filtered_sensors.append(wios_sensor)

print('Filtered sensors')
print(filtered_sensors)
print(len(filtered_sensors))
	

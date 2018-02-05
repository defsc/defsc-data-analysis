import json
import requests

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

def read_config(config_json_file_path):
	with open(config_json_file_path) as json_data_file:
	    data = json.load(json_data_file)
	return data

config = read_config('../config/config.json')

max_stations = 200
min_stations = 10

bbox_min_lat = 49.95
bbox_min_lon = 19.71
bbox_max_lat = 50.11
bbox_max_lon = 20.11

headers = {}

address = config['wunderground_sensors_url'].format(
		max_stations,
		min_stations,
		bbox_min_lat,
		bbox_min_lon,
		bbox_max_lat,
		bbox_max_lon)

response_json_string = execute_request(address, headers).decode('utf-8')
response_json_string = response_json_string.replace('__ng_jsonp__.__req3.finished(\n', '')
response_json_string = response_json_string.replace('\n\n);\n', '')

response_json = json.loads(response_json_string)

print(response_json)
print(len(response_json['stations']))

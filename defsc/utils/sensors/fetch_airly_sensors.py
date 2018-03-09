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

    # print(r.status_code)
    # print(r.headers)
    # print(r.content)

    return r.content


bbox_north_east_lat = 50.11
bbox_north_east_lon = 20.11
bbox_south_west_lat = 49.95
bbox_south_west_lon = 19.71

headers = {}

address = 'https://airapi.airly.eu/v1/sensors/current?northeastLat={}&northeastLong={}&southwestLong={}&southwestLat={}'.format(
    bbox_north_east_lat,
    bbox_north_east_lon,
    bbox_south_west_lon,
    bbox_south_west_lat)

headers['apikey'] = < AIRLY_APIKEY >

response_json_string = execute_request(address, headers)
response_json = json.loads(response_json_string.decode('utf-8'))

print(response_json)
print(len(response_json))

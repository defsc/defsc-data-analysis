from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib
import urllib
from PIL import Image
import numpy as np
import csv
from math import sin, cos, sqrt, atan2, radians
import PIL
from pandas import *
import os
import os.path

def get_distance(point1, point2):
    R = 6373.0

    lat1 = radians(point1[0])
    lon1 = radians(point1[1])
    lat2 = radians(point2[0])
    lon2 = radians(point2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c
#
def read_sensors_data():
    filename = "./airly_sensors.csv"
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        header = reader.__next__()
        values = {}
        for h in header:
            values[h] = []

        for row in reader:
            for h, v in zip(header, row):
                values[h].append(v)

    return values

def extract_corner_values(sensors_data):
    bl_lon = 181
    bl_lat = 91
    ur_lon = -181
    ur_lat = -91

    for lat in map(float, sensors_data['latitude']):
        if lat > ur_lat:
            ur_lat = lat
        if lat < bl_lat:
            bl_lat = lat

    for lon in map(float, sensors_data['longitude']):
        if lon > ur_lon:
            ur_lon = lon
        if lon < bl_lon:
            bl_lon = lon

    lat_difference = ur_lat - bl_lat
    bl_lat -= (0.1 * lat_difference)
    ur_lat += (0.1 * lat_difference)

    lon_difference = ur_lon - bl_lon
    bl_lon -= (0.1 * lon_difference)
    ur_lon += (0.1 * lon_difference)

    return bl_lat, bl_lon, ur_lat, ur_lon

def mark_point_on_map(sensors_data, point_id, map, value, cmap):
    id = -1

    for idx in range(len(sensors_data['id'])):
        if sensors_data['id'][idx] == point_id:
            id = idx
            break

    if id == -1:
        print("No LOM found")
        return -1
    lat = sensors_data['latitude'][id]
    lon = sensors_data['longitude'][id]

    norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)


    x, y = map(lon, lat)
    map.plot(x, y, 'bo', markersize=12, color=cmap(norm(value)))

def plot_values_on_stations(point_to_value, title, dir, new_filename):
    sensors_data = read_sensors_data()
    bl_lat, bl_lon, ur_lat, ur_lon = extract_corner_values(sensors_data)

    point1 = [ur_lat, ur_lon]
    point2 = [bl_lat, ur_lon]
    point3 = [bl_lat, bl_lon]
    point4 = [ur_lat, bl_lon]

    width = (get_distance(point1, point4) + get_distance(point2, point3))/2.0*1000
    height = (get_distance(point1, point2) + get_distance(point3, point4))/2.0*1000

    c_lat = (bl_lat + ur_lat)/2
    c_lon = (bl_lon + ur_lon)/2

    fig = plt.figure(num=None, figsize=(12, 8))
    m = Basemap(width=width,height=height,resolution='h',projection='aea', lon_0=c_lon,lat_0=c_lat)
    url = "http://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/export?bbox=" + \
          str(bl_lon) + "," + str(bl_lat) + "," + str(ur_lon) + "," + str(ur_lat) + \
          "&bboxSR=4326&imageSR=92176&size=3500,2500&dpi=50000&format=png32&f=image"

    if os.path.exists("map.bmp"):
        im = Image.open("map.bmp")
    else:
        im = Image.open(urllib.request.urlopen(url)).transpose(PIL.Image.FLIP_TOP_BOTTOM)
        im.save("map.bmp")
    m.imshow(im)

    cmap = plt.cm.seismic
    for lom, value in point_to_value.items():
        mark_point_on_map(sensors_data, lom, m, value, cmap)


    norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm)

    plt.title(title)

    if not os.path.exists("./results/" + dir):
        os.makedirs("./results/" + dir)

    plt.savefig("./results/" + dir + "/" + new_filename + ".png")
    plt.close()

def generate_list_of_correlations_to_check(dir):
    columns = set()

    for filename in os.listdir(dir):
        if filename == 'pollution.csv':
            continue

        csv_values = os.path.join(directory, filename)
        df = read_csv(csv_values, header=0, index_col=0)
        df.index = to_datetime(df.index)
        df = df.apply(lambda ts: ts.interpolate(method='nearest'))
        df = df.apply(lambda ts: ts.resample('1H').nearest())

        months = {n: g
                  for n, g in df.groupby(TimeGrouper('M'))}

        sorted_months = sorted(months.keys())
        for k in sorted_months:
            df_per_month = months[k]

            for col in df_per_month.columns:
                columns.add(col)
    correlations_to_check = []
    for p1 in columns:
        for p2 in columns:
            if not (p2, p1) in correlations_to_check:
                correlations_to_check.append((p1, p2))
    print(len(correlations_to_check))
    return correlations_to_check

if __name__ == "__main__":
    directory = '../data'
    correlations_to_check = generate_list_of_correlations_to_check(directory)
    # correlations_to_check = [['ow-hum', 'airly-pm25']]
    global_correlations = {}

    for filename in os.listdir(directory):
        if filename == 'pollution.csv':
            continue
        lom_id = filename[4:-4]

        csv_values = os.path.join(directory, filename)
        df = read_csv(csv_values, header=0, index_col=0)
        df.index = to_datetime(df.index)
        df = df.apply(lambda ts: ts.interpolate(method='nearest'))
        df = df.apply(lambda ts: ts.resample('1H').nearest())

        months = {n: g
                  for n, g in df.groupby(TimeGrouper('M'))}

        correlations = {}

        sorted_months = sorted(months.keys())
        for k in sorted_months:
            df_per_month = months[k]

            for pair_of_columns in correlations_to_check:
                if not pair_of_columns[0] in df_per_month.columns or not pair_of_columns[1] in df_per_month.columns:
                    continue
                correlation = df_per_month[pair_of_columns[0]].corr(df_per_month[pair_of_columns[1]])
                if np.isnan(correlation) or abs(correlation) < 0.3:
                    correlation = 0
                if (pair_of_columns[0], pair_of_columns[1]) not in correlations:
                    correlations[(pair_of_columns[0], pair_of_columns[1])] = [correlation]
                else:
                    correlations[(pair_of_columns[0], pair_of_columns[1])].append(correlation)

        months_labels = []
        for month_sample in sorted_months:
            months_labels.append(str(month_sample.year) + "-" + str(month_sample.month))

        for parameters, corrs in correlations.items():
            if not parameters in global_correlations:
                global_correlations[parameters] = {}
            for corr, month in zip(corrs, months_labels):
                if not month in global_correlations[parameters]:
                    global_correlations[parameters][month] = {}
                global_correlations[parameters][month][lom_id] = corr

    counter = 1
    for parameters, months in global_correlations.items():
        print(str(counter) + "/" + str(len(correlations_to_check)))
        month_nr = 1
        for month, values in months.items():
            print("\t" + str(month_nr) + "/" + str(len(months)))
            plot_values_on_stations(values, "Correlation of " + str(parameters) + " in " + month, parameters[0] + "|" + parameters[1] ,month)
            month_nr += 1
        counter += 1

# plot_values_on_stations({'210':0.7}, "tt", "n1")
# plot_values_on_stations({'218':}, "tt", "n2")
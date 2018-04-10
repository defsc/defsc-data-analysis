import os
from pandas import read_csv
from pandas import to_datetime
import matplotlib.pyplot as pyplot
import itertools


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

time_series_dict = {}

for filename in itertools.chain(listdir_fullpath('./filtered_csv/'), listdir_fullpath('./processed_csv_2017_2018/')): 
	df = read_csv(filename, header=0, index_col=0)
	df.index = to_datetime(df.index)

 	columns = df.columns

 	for column in columns:
 		if df[column].isnull().sum()/float(df[column].size) > 0.7:
 			df = df.drop(column, axis=1)

 	columns = df.columns

 	for column in columns:
		if column in time_series_dict:
			time_series_dict[column] = time_series_dict[column].append(df[column])
		else:
			time_series_dict[column] = df[column]

for key, value in sorted(time_series_dict.iteritems()):
	#print(key)
	#value.plot()
	#pyplot.title(key)
	#pyplot.show()

	value.to_csv('./merged_old_2017_2018/'  + key + '.csv', header=True)


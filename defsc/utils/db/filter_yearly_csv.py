import os
from pandas import read_csv
from pandas import to_datetime
import matplotlib.pyplot as pyplot

def standardie_unit(value):
	if '.' in value:
		new_value = float(value)
	else:
		new_value = float(value) / 1000.0

	return new_value


time_series_dict = {}

for filename in os.listdir('./csv/'):
	filename_prefix = filename.replace('.csv','')
	splitted_filename_prefix = filename_prefix.split('_')
	year = splitted_filename_prefix[0]
	var_name = splitted_filename_prefix[1]
	frequency = splitted_filename_prefix[2]

	print(year,var_name,frequency)

	csv = os.path.join('./csv/', filename)
	df = read_csv(csv, header=0, skiprows=[1,2], index_col=0)
	df.index = to_datetime(df.index)

	map_of_column_names = {column: column.replace('MpSkawinWIOSOsie0606','MpSkawOsOgro') for column in df.columns.tolist()}
	df.rename(columns = map_of_column_names, inplace=True)
	map_of_column_names = {column: column.replace('MpKrakowWIOSAKra6117','MpKrakAlKras') for column in df.columns.tolist()}
	df.rename(columns = map_of_column_names, inplace=True)
	map_of_column_names = {column: column.replace('MpKrakowWIOSBulw6118','MpKrakBulwar') for column in df.columns.tolist()}
	df.rename(columns = map_of_column_names, inplace=True)
	map_of_column_names = {column: column.replace('MpKrakowWIOSBuja6119','MpKrakBujaka') for column in df.columns.tolist()}
	df.rename(columns = map_of_column_names, inplace=True)
		
	if year != '2016':
		map_of_column_names = {column: column + '-' + var_name for column in df.columns.tolist()}
		df.rename(columns = map_of_column_names, inplace=True)
	else:
		map_of_column_names = {column: column.replace('-1g','') for column in df.columns.tolist()}
		df.rename(columns = map_of_column_names, inplace=True)

 	columns = filter(lambda column: any(x in column for x in ['MpKra','MpKasz','MpSkaw']), df.columns.tolist())    

 	for column in columns:
 		if df[column].dtype == object:
			df[column] = df[column].apply(lambda x: str(x).replace(',','.'))
 		
 		if year == '2016':
 			df[column] = df[column].apply(lambda x: standardie_unit(x))

 		df[column] = df[column].astype(float)

 		if column in time_series_dict:
 			time_series_dict[column] = time_series_dict[column].append(df[column])
 		else:
 			time_series_dict[column] = df[column]

 		#print(time_series_dict[column].head(5))

for key, value in sorted(time_series_dict.iteritems()):
	print(key)
	value.plot()
	pyplot.title(key)
	pyplot.show()

	value.to_csv('./filtered_csv/' + key + '.csv', header=True)

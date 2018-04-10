import os

template_dict = {}

template_dict['tlenek węgla'] = '-CO'
template_dict['benzen'] = '-C6H6'
template_dict['dwutlenek azotu'] = '-NO2'
template_dict['tlenki azotu'] = '-NOx'
template_dict['ozon'] = '-O3'
template_dict['pył zawieszony PM10'] = '-PM10'
template_dict['pył zawieszony PM2.5'] = '-PM2.5'
template_dict['dwutlenek siarki'] = '-SO2'

template_dict['Kaszów - '] = 'MpKaszowLisz'
template_dict['Kraków, Aleja Krasińskiego - '] = 'MpKrakAlKras'
template_dict['Kraków, ul. Bujaka - '] = 'MpKrakBujaka'
template_dict['Kraków, ul. Bulwarowa - '] = 'MpKrakBulwar'
template_dict['Kraków, ul. Dietla - '] = 'MpKrakDietla'
template_dict['Kraków, os. Piastów - '] = 'MpKrakOsPias'
template_dict['Kraków, ul. Złoty Róg - '] = 'MpKrakZloRog'
template_dict['Skawina, os. Ogrody - '] = 'MpSkawOsOgro'
template_dict['Kraków, os. Wadów - '] = 'MpKrakOsWad'
template_dict['Kraków, ul. Telimeny - '] = 'MpKrakTel'

template_dict[';'] = ','

for filename in os.listdir('./csv_2017_2018/'):
	filename_prefix = filename.replace('.csv','')

	csv = os.path.join('./csv_2017_2018/', filename)

	with open(csv, 'r') as file :
	  filedata = file.read()

	for key, value in template_dict.items():
		filedata = filedata.replace(key, value)

	# Write the file out again
	with open('./processed_csv_2017_2018/processed_' + filename, 'w') as file:
	  file.write(filedata)
# Creation of completely new dataset
* downloading the yearly data
* extracting the files ```unzip \*.zip```
* removing files with data of frequency other than 1h
* removing files containing the word "depozycja" in the name
* convnverting .xlsx files to .csv ```for i in *.xlsx; do  sudo libreoffice --convert-to csv "$i" ; done```
* deleting the first row from the ```2012_NOx_1g.csv``` file
* performing data transformation using ```filter_yearly_csv.py```
* downloading the monthly data
* removing columns with data of frequency other than 1h (PM10 with 24h frequency)
* modification of headers, standardization of column names using ```chagne_headers_of_monthly_csv.py``` (python3)
* inserting documents into the MongoDB database using ```insert_wios_measurements_to_mongo.py```

# Addition of data to the existing dataset
* downloading the data
* swapping custom names based on the dictionary in the file ```chagne_headers_of_monthly_csv.py```
* removing columns with data of frequency other than 1h (PM10 with 24h frequency)
* inserting documents into the MongoDB database with proper structure ```insert_wios_measurements_to_mongo.py```

# Generation of time-series .csv files based on data in MongoDB database
* run ```generate_ts_csv_based_on_mongodb_collection.py```

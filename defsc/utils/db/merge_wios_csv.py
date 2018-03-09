from pandas import read_csv, to_datetime
from matplotlib import pyplot

csv_1 = 'zloty-rog-gios-pjp-data.csv'
csv_2 = 'zloty-rog-luty-gios-pjp-data.csv'
merged_csv = 'merged-zloty-rog-gios-pjp-data.csv'

df_1 = read_csv(csv_1, header=[0, 1], index_col=0, sep=';')
df_2 = read_csv(csv_2, header=[0, 1], index_col=0, sep=';')

print('First data frame')
print(df_1.head())
print('Second data frame')
print(df_2.head())

df_1.index = to_datetime(df_1.index)
df_2.index = to_datetime(df_2.index)

merged_df = df_1.append(df_2)

print(merged_df)

merged_df.to_csv(merged_csv)

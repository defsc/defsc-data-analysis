from pandas import read_csv, to_datetime
from matplotlib import pyplot

time_series_csv = 'data/raw-204.csv'
df = read_csv(time_series_csv, header=0, index_col=0)
df.index = to_datetime(df.index)

for time_series in df.columns:
    df[time_series].dropna().plot()
    pyplot.title(time_series)
    pyplot.show()



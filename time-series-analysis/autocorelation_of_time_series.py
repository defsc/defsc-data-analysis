from matplotlib import pyplot
from pandas import read_csv, to_datetime
from pandas.tools.plotting import autocorrelation_plot

def plot_autocorelation_of_time_series(df, time_series_name):
    autocorrelation_plot(df[time_series_name].dropna())
    pyplot.title('{} autocorrelation chart'.format(time_series_name))
    pyplot.show()

time_series_csv = 'data/raw-204.csv'
df = read_csv(time_series_csv, header=0, index_col=0)
df.index = to_datetime(df.index)

plot_autocorelation_of_time_series(df, 'airly-pm10')

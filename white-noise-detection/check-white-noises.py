from pandas import read_csv, to_datetime
from pandas.tools.plotting import autocorrelation_plot
from matplotlib import pyplot
import os
import re

def plot_autocorrelation_for_timeseries(series, title):
    fig = pyplot.figure()
    fig.suptitle(title)
    autocorrelation_plot(series)
    pyplot.show()

file_pattern = re.compile("raw-204\.csv")
column_pattern = re.compile("airly.*")

base_path = os.environ['MSC_DATA'] + '/time_series_csv/'
for path, subdirs, files in os.walk(base_path):
    for name in files:
        full_path_for_ts_csv = os.path.join(path, name)
        csv_subpath = full_path_for_ts_csv[len(base_path):]

        if file_pattern.match(csv_subpath):
            df = read_csv(full_path_for_ts_csv, header=0, index_col=0)
            df.index = to_datetime(df.index)

            for time_series_name in df.columns:
                if column_pattern.match(time_series_name):
                    series = df[time_series_name].dropna()
                    plot_autocorrelation_for_timeseries(series, time_series_name + " for " + csv_subpath)
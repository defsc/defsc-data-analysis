import numpy as np

from matplotlib import pyplot
from pandas import read_csv, to_datetime

def crosscorr(datax, datay, lag=0):
    return datax.corr(datay.shift(lag))

def plot_cross_corelation_between_all_parameters_accross_the_time(df, predicted_parameter, lag):
    number_of_plot_rows = np.floor(len(df.columns.values)**0.5).astype(int)
    number_of_plot_cols = np.ceil(1.*len(df.columns.values)/number_of_plot_rows).astype(int)

    fig = pyplot.figure()
    i = 1

    for col_name in df.columns.values:
        correlation_coefficients_history = []
        for inner_lag in range(lag):
            correlation_coefficients_history.append(crosscorr(df[predicted_parameter], df[col_name], inner_lag))
        ax = fig.add_subplot(number_of_plot_rows, number_of_plot_cols, i)
        ax.plot(correlation_coefficients_history)
        ax.set_title(col_name)
        i+=1

    pyplot.show()

time_series_csv = 'data/raw-204.csv'
df = read_csv(time_series_csv, header=0, index_col=0)
df.index = to_datetime(df.index)
df = df.apply(lambda x: x.dropna())
df = df.apply(lambda x: x.resample('1H').mean())
print(df.tail(n=20))
plot_cross_corelation_between_all_parameters_accross_the_time(df, 'airly-pm1', 24)

# https://github.com/2wavetech/How-to-Check-if-Time-Series-Data-is-Stationary-with-Python
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller


def check_if_time_series_stationary_using_lag_plot(ts):
    lag_plot(ts)

    plt.show()


def check_if_time_series_stationary_using_autocorrelation(ts):
    autocorrelation_plot(ts)
    plt.show()


def check_if_time_series_stationary_using_sm_autocorrelation(ts):
    plot_acf(ts, lags=range(0, 200))
    plt.show()


def check_if_time_series_stationary_using_sm_partial_autocorrelation(ts):
    plot_pacf(ts, lags=range(0, 200))
    plt.show()


def time_series_historgram(ts):
    ts.hist()
    plt.show()


def check_if_time_series_is_stationary_by_spliting(ts, number_of_parts):
    splitted = np.array_split(ts.values, number_of_parts)

    for chunk in splitted:
        print('Mean:{}, variance{}'.format(chunk.mean(), chunk.var()))


def check_stationarity_using_fuller_test(ts, ts_name):
    result = adfuller(ts.values)
    print('%s \nADF Statistic: %f' % (ts_name, result[0]))
    print('p-value: %f' % result[1])
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


def print_number_of_nan_values(df):
    print('Number of rows of dataframe')
    print(len(df.index))
    print('Number of nan for specific time-series')
    print(df.isnull().sum())

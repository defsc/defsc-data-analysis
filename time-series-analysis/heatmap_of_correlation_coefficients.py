import numpy as np
import seaborn as sns

from matplotlib import pyplot
from pandas import read_csv, to_datetime


def crosscorr(datax, datay, lag=0):
    return datax.corr(datay.shift(lag))

def heat_map_of_correlation_coefficients(df):
    corr_df = df.corr(method='pearson')

    sns.heatmap(corr_df, annot=True)
    pyplot.show()

time_series_csv = 'data/raw-204.csv'
df = read_csv(time_series_csv, header=0, index_col=0)
df.index = to_datetime(df.index)
df = df.apply(lambda x: x.resample('1H').mean())
heat_map_of_correlation_coefficients(df)

import statsmodels.api as sm

from pandas import read_csv, to_datetime
from matplotlib import pyplot

time_series_csv = 'data/raw-204.csv'
df = read_csv(time_series_csv, header=0, index_col=0)
df.index = to_datetime(df.index)

for column in df.columns:
    ts = df[column]
    ts = ts.dropna()
    ts = ts.resample('1H').mean()
    ts = ts.fillna(ts.bfill())
    decomposition = sm.tsa.seasonal_decompose(ts, model='additive')
    fig = decomposition.plot()
    fig.suptitle(ts.name)
pyplot.show()


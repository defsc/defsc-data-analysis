import itertools
import warnings
import statsmodels.api as sm

from math import sqrt
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot
from pandas import read_csv, to_datetime, rolling_mean
from sklearn.metrics import mean_squared_error


def make_arima_forecasts(df, predicted_param_name):
    X = df[predicted_param_name]
    size = int(len(X) * 0.95)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()

    for t in range(0, len(test), 24):
        print(t, len(test))
        model = ARIMA(endog=np.asarray(history), order=(0, 1, 1))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast(steps=24)
        yhat = output[0]
        predictions = predictions + yhat.tolist()
        obs = test[t:t + 24]
        history = history + yhat.tolist()
        print('predicted={}, expected={}'.format(yhat, obs))
    # error = mean_squared_error(test, predictions)
    # print('Test MSE: %.3f' % error)
    # plot
    pyplot.plot(test.values)
    pyplot.plot(predictions, color='red')
    pyplot.show()


def grid_serach(old_ts):
    # Define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(0, 6)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 24) for x in list(itertools.product(p, d, q))]

    warnings.filterwarnings("ignore")  # specify to ignore warning messages

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                rmse = 0.0
                for i in range(1, 10, 1):
                    ts = old_ts[:-i * 24]

                    if i == 1:
                        obseved = old_ts[-i * 24:]
                    else:
                        obseved = old_ts[-i * 24:(-i + 1) * 24]

                    mod = sm.tsa.statespace.SARIMAX(ts,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)

                    results = mod.fit(disp=0)

                    pred_uc = results.get_forecast(steps=24)
                    rmse += sqrt(mean_squared_error(obseved.values, pred_uc.predicted_mean.values))
                print('ARIMA{}x{}24 - RMSE:{}'.format(param, param_seasonal, rmse))
            except:
                continue


time_series_csv = 'our_pollution_without_smothing.csv'
df = read_csv(time_series_csv, header=0, index_col=0)
df.index = to_datetime(df.index)
# arima_forest_forecasts = make_arima_forecasts(df, 'pm1')


ts = df['pm1'].resample('H').mean()
ts = ts.fillna(ts.bfill())
decomposition = sm.tsa.seasonal_decompose(ts, model='additive')
ts = decomposition.trend + decomposition.resid
ts = ts.dropna()
ts = ts.diff()
ts.plot(figsize=(15, 6))
pyplot.show()
print(ts.head())

old_ts = ts

grid_serach(old_ts)

# ax = old_ts.plot(label='observed', figsize=(20, 15))
# rmse = 0.0
# for i in range(1,10,1):
#     ts = old_ts[:-i * 24]
#     mod = sm.tsa.statespace.SARIMAX(ts,
#                                     order=(0,0,1),
#                                     seasonal_order=(0, 1, 0, 24),
#                                     enforce_stationarity=False,
#                                     enforce_invertibility=False)
#
#     results = mod.fit()
#
#     # Get forecast 500 steps ahead in future
#     pred_uc = results.get_forecast(steps=24)
#
#     # Get confidence intervals of forecasts
#     pred_ci = pred_uc.conf_int()
#     pred_uc.predicted_mean.plot(ax=ax)
#     ax.fill_between(pred_ci.index,
#                     pred_ci.iloc[:, 0],
#                     pred_ci.iloc[:, 1], color='k', alpha=.25)
#
#     if i == 1:
#         obseved = old_ts[-i * 24:]
#     else:
#         obseved = old_ts[-i * 24:(-i+1) * 24]
#
#     print('Predicted:')
#     print(len(pred_uc.predicted_mean.values))
#     print(pred_uc.predicted_mean)
#     print('Observed:')
#     print(len(obseved.values))
#     print(obseved)
#
#     rmse += sqrt(mean_squared_error(obseved.values, pred_uc.predicted_mean.values))
# print('RMSE: %f', rmse)
#
# ax.set_xlabel('Date')
# ax.set_ylabel('PM1 Levels')
# pyplot.show()

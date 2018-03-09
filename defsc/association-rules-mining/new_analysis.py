import numpy as np
import seaborn as sns
import statsmodels.api as sm

from pandas import concat
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot
from pandas import read_csv
from pandas.tools.plotting import autocorrelation_plot
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from statsmodels.tsa.arima_model import ARIMA

def read_time_series_from_csv(input_data_filename):
    return read_csv(input_data_filename, header=0, index_col=0)

def plot_data_frame_of_time_series(df):
    df.plot(subplots=True)
    pyplot.show()

def filter(df):
    return df

def fft_of_forecasted_time_series(df, predicted_param_name):
    soi = df[predicted_param_name].values
    n = len(soi)
    Y = np.fft.fft(soi)/n
    Y = Y[range(int(n/2))]

    timestep = 1.0
    frq = np.fft.fftfreq(n, d=timestep)
    frq = frq[range(int(n/2))]

    fig, ax = pyplot.subplots(2, 1)
    ax[0].plot(soi)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel(predicted_param_name)
    ax[1].plot(frq, abs(Y), 'r') # plotting the spectrum
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')

    pyplot.show()

def autocorelation_of_forecasted_time_series(df):
    autocorrelation_plot(df.pm1)
    pyplot.title('soi autocorrelation chart')
    pyplot.show()

def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation.
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))

def plot_cross_corelation_between_all_parameters_accross_the_time(df, predicted_parameter, lag):
    a = np.floor(len(df.columns.values)**0.5).astype(int)
    b = np.ceil(1.*len(df.columns.values)/a).astype(int)

    fig = pyplot.figure()

    i = 1
    for col_name in df.columns.values:
        correlation_coefficients = []
        for inner_lag in range(lag):
            correlation_coefficients.append(crosscorr(df[predicted_parameter],  df[col_name] ,inner_lag))
        ax = fig.add_subplot(a,b,i)
        ax.plot(correlation_coefficients)
        ax.set_title(col_name)
        i+=1

    pyplot.show()


def heat_map_of_correlation_coefficients(df):
    corr_df = df.corr(method='pearson')

    sns.heatmap(corr_df, annot=True)
    pyplot.show()

################################# Forecasting

def move_predicted_var_colum_to_end(df, predicted_param_name):
    cols_at_end = [predicted_param_name]
    return df[[c for c in df if c not in cols_at_end] + [c for c in cols_at_end if c in df]]

def series_to_supervised(df, predicted_param_name, n_in=1, n_out=1, dropnan=True):
    df = move_predicted_var_colum_to_end(df, predicted_param_name)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(len(df.columns))]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i).iloc[:,-1])
        if i == 0:
            names += ['VAR(t)']
        else:
            names += ['VAR(t+%d)' % i]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def forecast_linear(model, X):
    # make forecast
    forecast = model.predict(X)
    # convert to array
    return [x for x in forecast[0, :]]

def make_forecasts(model, test, n_ahead):
    forecasts = list()
    for i in range(len(test)):
        X = test[i, :-n_ahead]
        # make forecast
        forecast = forecast_linear(model, X)
        # store the forecast
        forecasts.append(forecast)
    return forecasts

def evaluate_forecasts(y, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = [row[i] for row in y]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        print('t+%d RMSE: %f' % ((i+1), rmse))

# plot the forecasts in the context of the original dataset, multiple segments
def     plot_forecasts(series, forecasts, n_test, xlim, ylim, n_ahead, linestyle = None):
    # plot the entire dataset in blue
    pyplot.figure()
    if linestyle==None:
        pyplot.plot(series, label='observed')
    else:
        pyplot.plot(series, linestyle, label='observed')
    pyplot.xlim(xlim, ylim)
    pyplot.legend(loc='upper right')
    # plot the forecasts in red
    X_axis = []
    Y_axis = []
    for i in range(len(forecasts)):
        if i%n_ahead ==0: # this ensures not all segements are plotted, instead it is plotted every n_ahead
               off_s = len(series) - n_test + 2 + i - 1
               off_e = off_s + len(forecasts[i]) + 1
               xaxis = [x for x in range(off_s, off_e)]
               yaxis = [series[off_s]] + forecasts[i]
               X_axis = X_axis + xaxis
               Y_axis = Y_axis + yaxis
               pyplot.plot(xaxis, yaxis, 'r')
               # print(off_s, off_e)
    # show the plot
    pyplot.show()

    pyplot.xlim(xlim, ylim)
    pyplot.plot(np.cumsum(series), label='observed')
    pyplot.plot(X_axis, np.cumsum(Y_axis), label='observed')
    pyplot.show()

    print(np.cumsum(series))
    print(np.cumsum(Y_axis))

# fit a linear model
def fit_linear(train, n_ahead):
    # reshape training into [samples, timesteps, features]
    X, Y = train[:, :-n_ahead], train[:, -n_ahead:]
    regr = linear_model.LinearRegression()
    model = MultiOutputRegressor(regr).fit(X, Y)
    return model

# fit a linear model
def fit_randomF(train, n_ahead):
    # reshape training into [samples, timesteps, features]
    X, Y = train[:, :-n_ahead], train[:, -n_ahead:]
    regr = RandomForestRegressor(max_depth = 30, random_state=2)
    model = MultiOutputRegressor(regr).fit(X, Y)
    return model

def make_arima_forecasts(df, predicted_param_name):
    X = df[predicted_param_name]
    size = int(len(X) * 0.95)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()

    for t in range(0,len(test),24):
        print(t, len(test))
        model = ARIMA(endog=np.asarray(history), order=(0,1,1))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast(steps=24)
        yhat = output[0]
        predictions = predictions + yhat.tolist()
        obs = test[t:t+24]
        history = history + yhat.tolist()
        print('predicted={}, expected={}'.format(yhat, obs))
    #error = mean_squared_error(test, predictions)
    #print('Test MSE: %.3f' % error)
    # plot
    pyplot.plot(test.values)
    pyplot.plot(predictions, color='red')
    pyplot.show()

def make_all_forecasts(df, train, test, ahead, lag):
    model = fit_linear(train, ahead)
    linear_regression_forecasts = make_forecasts(model, test, ahead)

    model = fit_randomF(train, ahead)
    random_forest_forecasts = make_forecasts(model, test, ahead)

    #arima_forest_forecasts = make_arima_forecasts(df, 'pm1')

    plot_forecasts(df['pm1'].values,linear_regression_forecasts , test.shape[0] + ahead - 1, 0, len(df['pm1']), ahead)
    plot_forecasts(df['pm1'].values,random_forest_forecasts , test.shape[0] + ahead - 1, 0, len(df['pm1']), ahead)
    #plot_forecasts(df['pm1'].values,arima_forest_forecasts , test.shape[0] + ahead - 1, 0, len(df['pm1']), ahead)

    plot_forecasts(df['pm1'].values,linear_regression_forecasts , test.shape[0] + ahead - 1, len(df['pm1']) - 200, len(df['pm1']), ahead, 'go')
    plot_forecasts(df['pm1'].values,random_forest_forecasts , test.shape[0] + ahead - 1, len(df['pm1']) - 200, len(df['pm1']), ahead, 'go')
    #plot_forecasts(df['pm1'].values,arima_forest_forecasts , test.shape[0] + ahead - 1, len(df['pm1']) - 100, len(df['pm1']), ahead, 'go')

    #print(df['pm1'].values.shape)
    #print(len(linear_regression_forecasts))
    #plot_forecasts(np.cumsum(df['pm1'].values).tolist(),np.cumsum(linear_regression_forecasts).tolist(), test.shape[0] + ahead - 1, 0, len(df['pm1']), ahead)


#time_series_csv = 'our_pollution.csv'
time_series_csv = 'our_pollution_without_smothing.csv'
df = read_time_series_from_csv(time_series_csv)
df = df.diff()
plot_data_frame_of_time_series(df)
df = filter(df)
fft_of_forecasted_time_series(df,'pm1')
autocorelation_of_forecasted_time_series(df)
plot_cross_corelation_between_all_parameters_accross_the_time(df, "pm1", 24)
heat_map_of_correlation_coefficients(df)

lag = 24
ahead = 24
reframed_df = series_to_supervised(df, 'pm1', lag, ahead)
print(reframed_df.head())
values = reframed_df.values
n_train = int(len(values) * 0.8)
train = values[:n_train, :]
test = values[n_train:, :]
make_all_forecasts(df, train, test, ahead, lag)

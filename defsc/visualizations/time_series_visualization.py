# Add visualization of forecasting results
import matplotlib.pyplot as pyplot
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from pandas.tools.plotting import autocorrelation_plot
from operator import sub
from operator import abs
from mpl_toolkits.mplot3d import axes3d


def crosscorr(datax, datay, lag=0):
    return datax.corr(datay.shift(lag))


def plot_autocorelation_of_time_series(df, time_series_name, save_to_file=False):
    autocorrelation_plot(df[time_series_name].dropna())
    pyplot.title('{} autocorrelation chart'.format(time_series_name))
    pyplot.show()


def plot_cross_corelation_between_all_parameters_accross_the_time(df, predicted_parameter, lag, save_to_file=False):
    number_of_plot_rows = np.floor(len(df.columns.values) ** 0.5).astype(int)
    number_of_plot_cols = np.ceil(1. * len(df.columns.values) / number_of_plot_rows).astype(int)

    fig = pyplot.figure()
    i = 1

    for col_name in df.columns.values:
        correlation_coefficients_history = []
        for inner_lag in range(lag):
            correlation_coefficients_history.append(crosscorr(df[predicted_parameter], df[col_name], inner_lag))
        ax = fig.add_subplot(number_of_plot_rows, number_of_plot_cols, i)
        ax.plot(correlation_coefficients_history)
        ax.set_title(col_name)
        i += 1

    pyplot.show()


def plot_all_time_series_from_dataframe(df, save_to_file=False):
    for time_series in df.columns:
        ts = df[time_series]
        pyplot.title(time_series)
        ts.plot()
        pyplot.show()


def plot_all_time_series_decomposition_from_dataframe(df, save_to_file=False):
    for column in df.columns:
        ts = df[column].dropna()

        decomposition = sm.tsa.seasonal_decompose(ts, model='additive')

        fig = decomposition.plot()
        fig.suptitle(ts.name)

    pyplot.show()


def plot_heat_map_of_correlation_coefficients(df, save_to_file=False):
    corr_df = df.corr(method='pearson')

    sns.heatmap(corr_df, annot=True)
    pyplot.show()

def plot_histograms_of_forecasts_errors_per_hour(y_true, y_pred, save_to_file=False):

    fig, ax = pyplot.subplots(nrows=5, ncols=5)
    fig.subplots_adjust(hspace=.5)

    col = ax[0][0]
    errors = map(sub, y_true.flatten(), y_pred.flatten())
    errors = list(map(abs, errors))
    col.hist(errors)
    col.set_title('Accumulated')


    for hour in range(y_true.shape[1]):
        col=ax[int((hour+1)/5)][(hour+1)%5]
        errors = map(sub, y_true[:, hour], np.asarray(y_pred)[:, hour])
        errors = list(map(abs, errors))
        col.hist(errors)
        col.set_title('Hour: {}'.format(hour+1))

    pyplot.show()

def plot_forecasting_result(y_real, y_pred, save_to_file=False):
    for prediction_step in range(y_real.shape[0]):
        flattened_y_true = y_real[prediction_step, :]
        flattened_y_pred = y_pred[prediction_step, :]

        if (any(flattened_y_true)):
            pyplot.plot(range(len(flattened_y_true)), flattened_y_true, range(len(flattened_y_pred)), flattened_y_pred, label=['true', 'pred'])
            pyplot.legend(['true', 'pred'], loc='upper center')
            pyplot.show()

def plot_forecasting_result_v2(y_real, y_pred, save_to_file=False):

    fig, ax = pyplot.subplots(nrows=6, ncols=4)
    fig.subplots_adjust(hspace=.6)

    for hour in range(24):
        col = ax[int(hour/4)][hour%4]
        flattened_y_true = y_real[:, hour]
        flattened_y_pred = y_pred[:, hour]
        col.plot(range(len(flattened_y_true)), flattened_y_true, range(len(flattened_y_pred)), flattened_y_pred)
        col.set_title('Hour: {}'.format(hour + 1))
        col.legend(['true', 'pred'], loc='upper right')

    pyplot.show()

def plot_forecast_result_in_3d(y_real, y_pred, save_to_file=False):
    X = np.empty([y_real.shape[0], 24])
    Y = np.empty([y_real.shape[0], 24])
    Z_real = np.empty([y_real.shape[0], 24])
    Z_pred = np.zeros([y_real.shape[0], 24])

    idx = 0
    for prediction_step in range(y_real.shape[0]):
        X[prediction_step,] = np.array([prediction_step + 1] * 24)
        Y[prediction_step,] = np.array(range(1,25))
        Z_real[prediction_step,] = np.array([y_real[prediction_step][0]] * 24)

        for prediction_hour in range(24):
            if prediction_step + prediction_hour < y_real.shape[0]:
                Z_pred[prediction_step + prediction_hour, prediction_hour] = y_pred[prediction_step, prediction_hour]

        if idx < 24:
            for prediction_hour in range(idx + 1, 24):
                Z_pred[prediction_step, prediction_hour] = Z_pred[prediction_step, idx]

        idx += 1


    fig = pyplot.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, abs(Z_real - Z_pred))
    ax.set_xlabel('Date (hours)')
    ax.set_ylabel('Hour of forecast')

    ax.set_zlabel('Absolute difference between predicted value and actual value')
    pyplot.show()
    #pyplot.savefig("third.png")

def plot_forecast_result_as_heat_map(y_real, y_pred, save_to_file=False):
    abs_diff = np.abs(y_real - y_pred)
    sns.heatmap(abs_diff, vmin=np.min(abs_diff), vmax=np.max(abs_diff), yticklabels=15)
    pyplot.show()


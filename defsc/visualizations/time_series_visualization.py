# Add visualization of forecasting results
import matplotlib.pyplot as pyplot
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from pandas.tools.plotting import autocorrelation_plot
from operator import sub
from operator import abs


def crosscorr(datax, datay, lag=0):
    return datax.corr(datay.shift(lag))


def plot_autocorelation_of_time_series(df, time_series_name):
    autocorrelation_plot(df[time_series_name].dropna())
    pyplot.title('{} autocorrelation chart'.format(time_series_name))
    pyplot.show()


def plot_cross_corelation_between_all_parameters_accross_the_time(df, predicted_parameter, lag):
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


def plot_all_time_series_from_dataframe(df):
    for time_series in df.columns:
        ts = df[time_series]
        pyplot.title(time_series)
        ts.plot()
        pyplot.show()


def plot_all_time_series_decomposition_from_dataframe(df):
    for column in df.columns:
        ts = df[column].dropna()

        decomposition = sm.tsa.seasonal_decompose(ts, model='additive')

        fig = decomposition.plot()
        fig.suptitle(ts.name)

    pyplot.show()


def plot_heat_map_of_correlation_coefficients(df):
    corr_df = df.corr(method='pearson')

    sns.heatmap(corr_df, annot=True)
    pyplot.show()

def plot_histograms_of_forecasts_errors_per_hour(y_true, y_pred):
    for hour in range(y_true.shape[1]):
        errors = map(sub, y_true[:, hour], np.asarray(y_pred)[:, hour])
        errors = list(map(abs, errors))
        pyplot.hist(errors)
        pyplot.title('Hour: {}'.format(hour))
        pyplot.show()

def plot_forecasting_result(y_real, y_pred):
    for prediction_step in range(y_real.shape[0]):
        flattened_y_true = y_real[prediction_step, :]
        flattened_y_pred = y_pred[prediction_step, :]

        if (any(flattened_y_true)):
            pyplot.plot(range(len(flattened_y_true)), flattened_y_true, range(len(flattened_y_pred)), flattened_y_pred)
            pyplot.show()

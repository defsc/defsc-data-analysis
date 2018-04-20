import os
import matplotlib.pyplot as pyplot
from pandas import read_csv, to_datetime
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
from defsc.data_structures_transformation.data_structures_transformation import transform_dataframe_to_supervised
from defsc.filtering.fill_missing_values import simple_fill_missing_values

def generate_arima_forecast(ts, number_of_timestep_ahead, p, d, q, percantage_of_train_data=0.8):
    number_of_rows = ts.size
    number_of_train_rows = int(number_of_rows * percantage_of_train_data)
    number_of_test_rows = number_of_rows - number_of_train_rows

    history = ts[:number_of_train_rows].values
    test = ts[number_of_train_rows:]
    predictions = []
    for i in range(number_of_test_rows):
        model = ARIMA(history, order=(p,d,q))
        model_fit = model.fit(disp=-1)
        prediction = model_fit.forecast(steps=number_of_timestep_ahead)[0]
        predictions.append(prediction)
        history = np.append(history, test[i])
    return np.asarray(predictions)

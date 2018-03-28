import os
import matplotlib.pyplot as pyplot
from pandas import read_csv, to_datetime
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
from defsc.data_structures_transformation.data_structures_transformation import transform_dataframe_to_supervised
from defsc.filtering.fill_missing_values import simple_fill_missing_values

def generate_arima_forecast(ts, number_of_timestep_ahead, percantage_of_train_data=0.8):
    number_of_rows = ts.size
    number_of_train_rows = int(number_of_rows * percantage_of_train_data)
    number_of_test_rows = number_of_rows - number_of_train_rows

    history = ts[:number_of_train_rows]
    test = ts[number_of_train_rows:]
    predictions = []
    for i in range(number_of_test_rows):
        model = ARIMA(history, order=(4,0,0))
        model_fit = model.fit(disp=-1)
        prediction = model_fit.forecast(steps=number_of_timestep_ahead)[0]
        predictions.append(prediction)
        history.append(test[i*number_of_timestep_ahead:(i+1)*number_of_timestep_ahead])
    return np.asarray(predictions)

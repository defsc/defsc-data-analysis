#based on http://www.nehalemlabs.net/prototype/blog/2013/04/05/an-introduction-to-smoothing-time-series-in-python-part-i-filtering-theory/
#import matplotlib

#matplotlib.use('Qt5Agg')

from defsc.filtering.fill_missing_values import simple_fill_missing_values

import numpy as np
from pandas import read_csv, to_datetime
from scipy.interpolate import UnivariateSpline
from scipy.signal import wiener, filtfilt, butter, gaussian, freqz
from scipy.ndimage import filters
import scipy.optimize as op
import matplotlib.pyplot as plt

def rmse(filteterd, measurement, npts):
    return np.sqrt(np.sum(np.power(filteterd-measurement,2)))/npts

def testGauss(measurement, npts):
    b = gaussian(500, 5)
    #ga = filtfilt(b/b.sum(), [1.0], y)
    ga = filters.convolve1d(measurement, b/b.sum())
    #plt.plot(range(npts), ga, label='gauss')
    print("gaerr", rmse(ga, measurement, npts))
    return ga

def testButterworth(measurement, npts):
    x = range(npts)
    b, a = butter(4, 0.8)
    fl = filtfilt(b, a, measurement)
    plt.plot(x,fl)
    print("flerr", rmse(fl, measurement, npts))
    return fl

def testWiener(measurement, npts):
    x = range(npts)
    wi = wiener(measurement)
    plt.plot(x,wi)
    print("wieerr", rmse(wi, measurement, npts))
    return wi

def testSpline(measurement, npts):
    x = range(npts)
    sp = UnivariateSpline(x, measurement, s=4)
    plt.plot(x,sp(x))
    print("splerr", rmse(sp(x), measurement, npts))
    return sp(x)

def plotPowerSpectrum(y, w):
    dt = 1
    ft = np.fft.rfft(y)
    ps = np.real(ft*np.conj(ft))*np.square(dt)
    plt.plot(w, ps)

if __name__ == "__main__":
    time_series_csv = '../data/raw-204.csv'
    df = read_csv(time_series_csv, header=0, index_col=0)
    df.index = to_datetime(df.index)

    df = simple_fill_missing_values(df)

    ts = df['airly-pm10']
    ts_values = ts.values
    ts_len = len(ts_values)

    plt.plot(range(ts_len),ts_values)
    #plt.plot(x,y,ls='none',marker='.')

    ga = testGauss(ts_values, ts_len)
    fl = testButterworth(ts_values, ts_len)
    wi = testWiener(ts_values, ts_len)
    sp = testSpline(ts_values, ts_len)

    plt.legend(['meas', 'gauss', 'iir', 'wie', 'spl'], loc='upper center')
    plt.show()

    w = np.fft.fftfreq(ts_len, d=1)
    w = np.abs(w[:int(ts_len/2)+1]) #only freqs for real fft
    plotPowerSpectrum(ts_values, w)
    plotPowerSpectrum(ga, w)
    plotPowerSpectrum(fl, w)
    plotPowerSpectrum(wi, w)
    plotPowerSpectrum(sp, w)
    plt.show()

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 08:34:15 2017

@author: H391066
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
ts_data = pd.read_csv('Time_Series_KPComp.csv', parse_dates=['period_dt'], index_col='period_dt',date_parser=dateparse)

print(ts_data.head())

print('\n Data Types:')
print(ts_data.dtypes)

from statsmodels.tsa.stattools import adfuller
#def test_stationarity(timeseries):
    
    #Determing rolling statistics
 #   rolmean = pd.rolling_mean(timeseries, window=12)
 #   rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
 #   orig = plt.plot(timeseries, color='blue',label='Original')
 #   mean = plt.plot(rolmean, color='red', label='Rolling Mean')
 #   std = plt.plot(rolstd, color='black', label = 'Rolling Std')
 #   plt.legend(loc='best')
 #   plt.title('Rolling Mean & Standard Deviation')
 #   plt.show(block=False)
    
    #Perform Dickey-Fuller test:
 #   print('Results of Dickey-Fuller Test:')
 #   dftest = adfuller(timeseries, autolag='AIC')
 #   dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
 #   for key,value in dftest[4].items():
 #       dfoutput['Critical Value (%s)'%key] = value
 #   print(dfoutput)
    
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
    
C_Diff_ts = ts_data.value[(ts_data.Measure_Name == 'C-diff per 10K Pt Days') & (ts_data.series_name == 'KP-Nat')]

test_stationarity(C_Diff_ts)



# Let's perform some differencing to eliminate trend/seasonality

C_Diff_ts_diff = C_Diff_ts - C_Diff_ts.shift()
plt.plot(C_Diff_ts_diff)

#Compare this
C_Diff_ts_diff.dropna(inplace = True)
test_stationarity(C_Diff_ts_diff)

#ARIMA Model
from statsmodels.tsa.arima_model import ARIMA
arima_model = ARIMA(C_Diff_ts, order = (2,1,2))
ARIMA_results = arima_model.fit(disp = -1)
plt.plot(C_Diff_ts_diff)
plt.plot(ARIMA_results.fittedvalues, color = 'green')
plt.title('RSS: %.4f'% sum((ARIMA_results.fittedvalues-C_Diff_ts_diff)**2))

predictions_ARIMA_diff = pd.Series(ARIMA_results.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())

#Change back to originals
predictions_ARIMA = pd.Series(C_Diff_ts.ix[0], index = C_Diff_ts.index)
predictions_ARIMA = predictions_ARIMA.add(predictions_ARIMA_diff_cumsum, fill_value = 0)
predictions_ARIMA.head()

plt.plot(C_Diff_ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-C_Diff_ts)**2)/len(C_Diff_ts)))


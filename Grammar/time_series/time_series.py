from os.path import dirname, abspath
import data.file.file_ops as file_ops
from settings.config import ROOT_PATH
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
from statsmodels.tsa.seasonal import seasonal_decompose

# data ops
datasets_path = ROOT_PATH + '/datasets'
file_name = 'AirPassengers.csv'
file = file_ops.File_ops(datasets_path, file_name)
data = file.read_csv_file()
ts = data['Passengers']
ts_log = np.log(ts)

print(data.head(5))

# plt.plot(ts_log)
# plt.show()

# stationary test
## rolling statistics
def rolling_statistics(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12, center=False).mean()
    rolstd = timeseries.rolling(window=12, center=False).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=True)

if False:
    rolling_statistics(ts)

## ADF test
from statsmodels.tsa.stattools import adfuller
def adf_test(timeseries):
    rolling_statistics(timeseries) # plot
    print('Results of Augment Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

if True:
    adf_test(ts)


## decompose
decomposition = seasonal_decompose(ts, model="additive", freq=12)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

if False:
    for i in range(len(trend)):
        print(round(trend[i], 2), '\t', round(seasonal[i], 2), '\t', round(residual[i], 2), '\t', ts[i])

if False:
    plt.subplot(411)
    plt.plot(ts,label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413);
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

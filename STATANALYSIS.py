from pandas import Series
from matplotlib import pyplot
from matplotlib import interactive
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
import numpy
import math
from pandas import Series
from statsmodels.graphics.tsaplots import plot_acf
from pandas.tools.plotting import autocorrelation_plot
from pandas.tools.plotting import lag_plot
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import log

time_interval = input('Give me a time_interval (10,15,20,30)? ')
print('You said: ' + str(time_interval))


forecasting_horizon = input('Give me the forecasting horizon in hours? ')
print('You said: ' + str(forecasting_horizon))

steps_ahead= (60.0/time_interval)*forecasting_horizon
print('You want: ' + str(steps_ahead))
  

#Quick Check for Autocorrelation
series = Series.from_csv('/home/aris/Desktop/Short-Term-Electric-Load-Forecasting-CSL-master/price.csv', header=0)
lag_plot(series,lag=300)
pyplot.show()

############################################
plot_acf(series, lags=300)
pyplot.show()
##############################################
series.hist()
pyplot.show()
###############################################
X = series.values
pyplot.plot(X)
pyplot.show()
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
	

split = len(X) / 2
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))

# split dataset
X = series.values
percentage= 1.00*(steps_ahead/len(X))
#train, test = X[1:len(X)-7], X[len(X)-7:]
test=int(round(len(X)*0.15))
print(len(X))
train, test = X[1:len(X)-test], X[len(X)-test+1:]
# train autoregression
model = AR(train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
errorRMSE = (math.sqrt(mean_squared_error(test, predictions)))#/(numpy.max(X)-numpy.min(X)))
errorMAE = mean_absolute_error(test, predictions)#/(numpy.max(X)-numpy.min(X))
accuracy=(1-errorRMSE)*100
print('Test RMSE: %.3f' % errorRMSE)
print('Test MAE: %.3f' % errorMAE)
print('Accuracy: %.5f  per cent' % (accuracy))

# plot results

pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

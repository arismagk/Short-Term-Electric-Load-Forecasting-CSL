# LSTM for international airline passengers problem with time step regression framing
import time
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)


time_interval = input('Give me a time_interval (10,15,20,30)? ')
print('You said: ' + str(time_interval))


forecasting_horizon = input('Give me the forecasting horizon in hours? ')
print('You said: ' + str(forecasting_horizon))

steps_ahead= (60.0/time_interval)*forecasting_horizon
print('You want: ' + str(steps_ahead))
 
# load the dataset
dataframe = read_csv('/home/aris/Documents/data/lamia/amfissa.csv', usecols=[1], engine='python', skipfooter=3)
X = dataframe.values

dataset = dataframe.values
dataset = dataset.astype('float32')
percentage= 1.0*(1-(steps_ahead/len(dataset)))
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets


train_size = int(len(dataset) * percentage-len(dataset)*0.25)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 47
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# create and fit the DNN network
start_time = time.time()

model = Sequential()
model.add(Flatten(input_shape=trainX.shape[1:]))
#model.add(Dense(4, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
print(numpy.max(trainY)) 
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))#/(numpy.max(X)-numpy.min(X))
print('Train Score: %.5f RMSE' % (trainScore))
trainScore1 = mean_absolute_error(trainY[0], trainPredict[:,0])#/(numpy.max(X)-numpy.min(X))
print('Train Score: %.5f MAE' % (trainScore1))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))#/(numpy.max(X)-numpy.min(X))
print('Test Score: %.5f RMSE' % (testScore))
testScore1 = mean_absolute_error(testY[0], testPredict[:,0])#/(numpy.max(X)-numpy.min(X))
print('Test Score: %.5f MAE' % (testScore1))
accuracy=(1-testScore)*100
print('Accuracy calculated with RMSE: %.5f  per cent' % (accuracy))
elapsed_time = time.time() - start_time

print('Runned in: %.2f seconds' % (elapsed_time))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
#testPredictPlot[len(trainPredict)+(look_back)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot,color='green')
plt.plot(testPredictPlot,color='red')
plt.show()


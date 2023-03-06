# solution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt

import plotly.graph_objects as go
from matplotlib import pyplot

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv('dataset/municipality_bus_utilization.csv')

def check_df(df, head=5):
    print('### Shape ###')
    print(df.shape)
    print('### Types ###')
    print(df.dtypes)
    print('### Head ###')
    print(df.head(head))
    print('### NA ###')
    print(df.isnull().sum())
    print('### Quantiles ###')
    print(df.describe([0, 0.05, 0.50, 0.95, 0.99, 1]))

check_df(df)

df.columns = df.columns.str.upper()
df.head()

df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
df.dtypes

# aggregation of two measurements for an hour by taking max value

# her saatte max usage:
df_max = df.groupby([pd.Grouper(key='TIMESTAMP', axis=0, freq='H')])['USAGE'].max()
df_max.head()


# aşağıdaki kod ile her belediye için belirtiyor max usage:
df_municipality = df.groupby([pd.Grouper(key='TIMESTAMP', axis=0, freq='H'), 'MUNICIPALITY_ID'])['USAGE'].max()
df_municipality.head()

# görselleştirme:
plt.figure(figsize=(10, 10))
sns.set(font_scale=0.7)
g = sns.barplot(x="MUNICIPALITY_ID", y="USAGE", data=df.sort_values(by="USAGE", ascending=False))
plt.title("TOTAL USAGE")
plt.show()

g = sns.lineplot(x='TIMESTAMP', y="USAGE", data=df, hue='MUNICIPALITY_ID')
plt.title("Total Amount")
plt.show()


df_municipality.head()

municipality_id = list(df['MUNICIPALITY_ID'].unique())
municipality_id.sort()
municipality_id
df_each = {i: df[df['MUNICIPALITY_ID'] == i] for i in municipality_id}

#skipping
df0 = df_each[0]
df1=df_each[1]
df2=df_each[2]
df3=df_each[3]
df4=df_each[4]
df5=df_each[5]
df6=df_each[6]
df7=df_each[7]
df8=df_each[8]
df9=df_each[9]

df0.head()
df.info()

df0max = df0.set_index('TIMESTAMP')
df0max = df0.resample('H', on='TIMESTAMP').max()

df0max.head()


df0max['year'] = df0max.index.year
df0max['month'] = df0max.index.month
df0max['day'] = df0max.index.day
df0max['hour'] = df0max.index.hour

df0max.head(20)

df0max = df0max[(df0max.hour >= 7) & (df0max.hour < 17)]
df0max.tail(20)

df_fill = df0max.fillna(method="bfill")
df_fill.isnull().sum()
df0max.isnull().sum()
df_fill.USAGE.plot()
df0max.USAGE.plot()

plt.figure(figsize=(10, 10))
df0max.USAGE.plot()
df_fill.USAGE.plot()


df0max = df0max.USAGE
df0max.interpolate().plot()
df0int = df0max.interpolate()

plt.figure(figsize=(10, 10))
df0max.plot()
df_fill.USAGE.plot()
df0int.plot()

df0 = df_fill.USAGE
df0 = pd.DataFrame(df0)
df0.plot()


#train-test

df0.loc['2017-08-05 07:00:00':]
df0.shape

df0.info()

train, test = df0.loc[:'2017-08-05 07:00:00'], df0.loc['2017-08-05 07:00:00':]
print(len(train), len(test))

train.head()
test.info()

plt.plot(train, color = "green")
plt.plot(test, color = "blue")
plt.ylabel('USAGE')
plt.xlabel('DATE')
plt.xticks(rotation=45)
plt.title("Train/Test split - Usage Data")
plt.show()

#Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

#time series generator
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
help(TimeseriesGenerator)

length = 10
batch_size = 10
generator = TimeseriesGenerator(train_scaled, train_scaled, length = length, batch_size = batch_size)
validation_generator = TimeseriesGenerator(test_scaled, test_scaled, length = length, batch_size = batch_size)

len(generator)
generator[0]

X, y = generator[0]
print(f'Given the Array: \n{X.flatten()}')
print(f'Predict this y: \n {y}')

#modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout

train_scaled.shape
n_features = train_scaled.shape[1]
n_features

model = Sequential()
model.add(LSTM(64, activation = 'relu', return_sequences=True, input_shape = (length, n_features)))
model.add(Dropout(0.2))
model.add(LSTM(32, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse')

model.summary()

#EarlyStopping and validation generator
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor = 'val_loss', patience = 10,restore_best_weights = True)
len(validation_generator)

model.fit(x = generator,
          epochs = 50,
          validation_data = validation_generator,
          callbacks = [early_stop])


loss_df = pd.DataFrame(model.history.history)
loss_df.plot()

# Evaluation on Test Data
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    score = r2_score(actual, pred)
    return print("r2_score:", score, "\nmae:", mae, "\nmse:",mse, "\nrmse:",rmse)


predictions_scaled = []

first_eval_batch = train_scaled[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(length):
    # get prediction 1 time stamp ahead
    current_pred = model.predict(current_batch)

    # store prediction
    predictions_scaled.append(current_pred[0])

    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:, 1:, :], [current_pred], axis=1)

predictions_scaled

eval_metrics(test_scaled[:length], predictions_scaled)
#r2_score: -0.27874720249075646
#mae: 0.10130786013012423
#mse: 0.013137527118604247
#rmse: 0.11461905216238812

#Inverse Transformation and Comparing

predictions = scaler.inverse_transform(predictions_scaled)
compare = test[:10]
compare['Predictions_tanh'] = predictions
compare

compare.plot()

eval_metrics(compare.USAGE, compare.Predictions_tanh)
#r2_score: -0.27874720249075646
#mae: 149.22647797167298
#mse: 28504.873577522074
#rmse: 168.8338638351977

#modelling Bidirectional with "relu" activation function - 2

model2 = Sequential()
model2.add(Bidirectional(LSTM(100, activation = 'relu', return_sequences=True, input_shape = (length, n_features))))
model2.add(Dropout(0.2))
model2.add(Bidirectional(LSTM(50, activation = 'relu', return_sequences=True)))
model2.add(Dropout(0.2))
model2.add(Bidirectional(LSTM(20, activation = 'relu')))
model2.add(Dropout(0.2))
model2.add(Dense(1))
model2.compile(optimizer = 'adam', loss = 'mse')

model2.summary()

early_stop = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)

model2.fit_generator(generator,
                    epochs = 50,
                    validation_data = validation_generator,
                    callbacks = [early_stop])

loss_df = pd.DataFrame(model2.history.history)
loss_df.plot()


#Evaluation on Test Data

predictions_scaled = []

first_eval_batch = train_scaled[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(length):
    # get prediction 1 time stamp ahead
    current_pred = model2.predict(current_batch)

    # store prediction
    predictions_scaled.append(current_pred[0])

    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:, 1:, :], [current_pred], axis=1)


#Inverse Transformation and Comparing

predictions = scaler.inverse_transform(predictions_scaled)
compare['Predictions_2'] = predictions
compare


compare.plot()
eval_metrics(compare.USAGE, compare.Predictions_2)

#Retrain and Forecasting
full_scaler = MinMaxScaler()
scaled_full_data = full_scaler.fit_transform(df0)

length=10
length
batch_size = 1
batch_size

generator = TimeseriesGenerator(scaled_full_data, scaled_full_data, length = length, batch_size = batch_size)
len(df0)

len(generator)

model = Sequential()
model.add(LSTM(20, activation = 'relu', return_sequences=True, input_shape = (length, n_features)))
model.add(Dropout(0.2))
model.add(LSTM(10, activation = 'relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(5, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse')

scaled_full_data.shape

scaled_full_data[-length:].shape


forecast = []

first_eval_batch = scaled_full_data[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(100):
    current_pred = model.predict(current_batch)

    forecast.append(current_pred[0])

    current_batch = np.append(current_batch[:, 1:, :], [current_pred], axis=1)


print(current_pred)

print(predictions_scaled)
print(current_batch)

forecast = scaler.inverse_transform(forecast)
forecast

df0


# week before last week
forecast_index = pd.date_range(start = '2017-08-05 07:00:00', periods = 100, freq = 'H')
forecast_index

forecast_df = pd.DataFrame(data = forecast, index = forecast_index, columns = ['Forecast'])
forecast_df


plt.figure(figsize = (16, 8))
plt.plot(df0.index, df0['USAGE'])
plt.plot(forecast_df.index, forecast_df['Forecast'])


ax = df0.plot()
forecast_df.plot(ax = ax, figsize = (16, 8))
plt.xlim('2017-08-05 07:00:00', '2017-08-20 07:00:00')


#last week
forecast_index = pd.date_range(start = '2017-08-12 07:00:00', periods = 100, freq = 'H')
forecast_index

forecast_df = pd.DataFrame(data = forecast, index = forecast_index, columns = ['Forecast'])
forecast_df

plt.figure(figsize = (16, 8))
plt.plot(df0.index, df0['USAGE'])
plt.plot(forecast_df.index, forecast_df['Forecast'])

ax = df0.plot()
forecast_df.plot(ax = ax, figsize = (16, 8));

ax = df0.plot()
forecast_df.plot(ax = ax, figsize = (16, 8))
plt.xlim('2017-08-12 07:00:00', '2017-08-20 00:00:00')




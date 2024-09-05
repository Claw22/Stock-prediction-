import pandas as pd
stock _data = pd.read_csv(" •/NFLX.csv,index_col='Date')
stock_data.head()
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import datetime as dt

plt.figure(figsize=(15,10))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter ('%Y-%m-%d') )
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=60))
x_dates = [dt. datetime.strptime(d, '%Y-%m-%d') .date() for d in stock__data. index.values]

plt.plot(x dates, stock _data[ 'High'], label='High')
plt. plot(x_dates, stock_data[ 'Low'], label='Low')
plt. xlabel('Time Scale')
plt.ylabel( 'Scaled USD')
plt.legend()
plt.gcf().autofmt_xdate()
plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensonflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import Earlystopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
12 from sklearn.metrics import mean_squared_error
13 from sklearn.metrics import mean_absolute_percentage_error
14 from sklearn model_selection import train_test_split
15 from sklearn model_selection import TimeSeriesSplit
16 from sklearn.metrics import mean_squared_error

target_y = stock _data[ 'close']
X_feat = stock_data.iloc[:,0:3]
#Feature Scaling
sc = StandardScaler()
X_ft = sc.fit_transform(X_feat.values)
X_ft = pd.DataFrame(columns=X_feat.columns,data=X_ft,index=X_feat.index)

def 1stm_split(data, n_steps) :
x, y = [], []
for i in range(len(data) -n_steps+1):
X.append (data [i:i + n_steps, : -1])
y.append (data [i + n_steps-1, -1])
return np.array(X), np.array(y)

x1, y1 = lstm_split(stock _data_ft. values, n_steps=2)
train_split=0.8
split_idx = int(np. ceil(len(X1)*train_split))
date_index = stock_data_ft.index
X_train, X_test = X1[:split_idx], X1[split_idx:]
y_train, y_test = y1[: split_idx], y1[split_idx: ]
X train_date, X_test_date = date_index[:split_idx], date_index[split_idx:]
print(X1.shape, X_train.shape, X_test.shape, _test.shape)

lstm = Sequential()
1stm.add LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]),
activation='relu', return_sequences=True))
lstm.add(Dense (1))
1stm.compile(loss= 'mean_squared_error', optimizer='adam')
Istm.summary()

history=lstm.fit(X_train, y_train,epochs=100, batch_size=4,
verbose=2, shuffle=False)
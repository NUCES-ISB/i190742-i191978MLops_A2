import pandas as pd
from pandas_datareader import data as web
import datetime as dt
import numpy as np
import matplotlib as plt
import yfinance as fyn


from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

from flask import Flask, render_template



crypto = 'BTC'
currency = 'USD'

start = dt.datetime(2016,1,1)
end = dt.datetime.now()

fyn.pdr_override()
data = web.get_data_yahoo(f'{crypto}-{currency}', start, end).reset_index()

print(data)

print(data.columns)

print((data['Date'][0].to_pydatetime().strftime('%Y-%m-%d')))

new_data = []

for i in range(0, len(data)):
    new_data.append((data['Date'][i].to_pydatetime().strftime('%Y-%m-%d'), data['Close'][i]))


print(new_data)


app = Flask(__name__)

@app.route("/")
def home():
    data = new_data

    labels = [row[0] for row in data]
    values = [row[1] for row in data]

    return render_template("graph.html", labels=labels, values=values)




# scaler = MinMaxScaler(feature_range=(0,1))
# scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

# days = 60

# x_train, y_train = [], []

# for x in range (days, len(scaled_data)):
#     x_train.append(scaled_data[x-days:x, 0])
#     y_train.append(scaled_data[x, 0])

# x_train, y_train = np.array(x_train), np.array(y_train)
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# model = Sequential()

# model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50))
# model.add(Dropout(0.2))
# model.add(Dense(units=1))

# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(x_train, y_train, epochs=25, batch_size=32)




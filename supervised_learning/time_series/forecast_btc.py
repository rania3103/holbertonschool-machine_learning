#!/usr/bin/env python3
"""Bitcoin (BTC) became a trending topic after its price peaked in 2018.
Many have sought to predict its value in order to accrue wealth.
Let’s attempt to use our knowledge of RNNs to attempt just that."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
# load data
df = pd.read_csv("combined_scaled_data.csv")
# features
X = df.drop(columns=['Close']).values
# target
y = df['Close'].values
# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# scale data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# fit and transform target
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()
# make sequences


def create_sequences(X, y, time_steps):
    sequences = []
    targets = []
    for i in range(len(X) - time_steps):
        sequences.append(X[i:i + time_steps])
        targets.append(y[i + time_steps])
    return np.array(sequences), np.array(targets)


# Set time_steps to 1440 (24 hours of 60-second windows)
time_steps = 1440
X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps)

X_train_seq = X_train_seq.reshape(
    X_train_seq.shape[0],
    X_train_seq.shape[1],
    X_train_seq.shape[2])
X_test_seq = X_test_seq.reshape(
    X_test_seq.shape[0],
    X_test_seq.shape[1],
    X_test_seq.shape[2])
# build model
model = tf.keras.Sequential([tf.keras.layers.LSTM(50, return_sequences=True,
                            input_shape=(X_train_seq.shape[1],
                                         X_train_seq.shape[2])),
                             tf.keras.layers.Dense(1)])
# compile model
model.compile(optimizer='adam', loss='mean_squared_error')
# fit model
model.fit(
    X_train_seq,
    y_train_seq,
    epochs=10,
    batch_size=8,
    validation_data=(
        X_test_seq,
        y_test_seq))
# calculate loss
loss = model.evaluate(X_test_seq, y_test_seq)
print(f'loss: {loss}')
# predictions
predictions = model.predict(X_test_seq)
model.save('btc_price_forecast_model.h5')
print("Model saved as 'btc_price_forecast_model.h5'")

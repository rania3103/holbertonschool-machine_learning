#!/usr/bin/env python3
"""train an LSTM model to predict BTC close price 1 hour ahead using past 24 hours of data."""

import numpy as np
import pandas as pd
import tensorflow as tf
import os
# disable GPU to avoid DNN init issuesAdd commentMore actions
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

seq_len = 1440
batch_size = 8
buff_size = 100
# load data
df = pd.read_csv('combined_preprocessed_data.csv')
# features & target
X_cols = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']
y_col = 'Close'
X = df[X_cols].values
y = df[y_col].values


def sequence_generator(data, targets, seq_length):
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    targets = tf.convert_to_tensor(targets, dtype=tf.float32)

    for i in range(len(data) - seq_length):
        yield data[i:i + seq_length], targets[i + seq_length]


# create dataset using generator
dataset = tf.data.Dataset.from_generator(
    lambda: sequence_generator(X, y, seq_len),
    output_types=(tf.float32, tf.float32),
    output_shapes=([seq_len, len(X_cols)], [])
)

# shuffle,batch and prefetch
dataset = dataset.shuffle(buff_size).batch(
    batch_size).prefetch(tf.data.AUTOTUNE)

# split dataset into train and test
total_samples = len(df) - seq_len
train_size = int(0.8 * total_samples)
dataset = dataset.enumerate()
train_dataset = dataset.filter(lambda i, _: i < train_size).map(lambda _, v: v)
test_dataset = dataset.filter(lambda i, _: i >= train_size).map(lambda _, v: v)

# build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=False, input_shape=(seq_len, len(X_cols))),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

# compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='mse',
              metrics=[tf.keras.metrics.MeanAbsoluteError()])

# callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_btc_model.h5', save_best_only=True)

# train model
print("training model:")
history = model.fit(
    train_dataset,
    epochs=20,
    validation_data=test_dataset,
    callbacks=[early_stop, checkpoint]
)

# evaluate on test set
loss, mae = model.evaluate(test_dataset)
print(f"test loss (MSE): {loss:.6f}, MAE: {mae:.6f}")

# save final model
model.save('final_btc_forecast_model.h5')
print("final model saved as 'final_btc_forecast_model.h5'")

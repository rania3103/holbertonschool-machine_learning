#!/usr/bin/env python3
"""a python script that optimizes a basic neural network model  using GPyOpt """
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
import GPyOpt
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# disable GPU to avoid DNN init issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def load_dataset_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)


def train_model(params):
    params = params[0]
    lr = float(params[0])
    units = int(round(params[1]))
    dropout = float(params[2])
    l2_lambda = float(params[3])
    batch_size = int(round(params[4]))
    (x_train, y_train), (x_val, y_val) = load_dataset_mnist()
    model = keras.Sequential([layers.Dense(units,
                                           activation="relu",
                                           input_shape=(784,
                                                        ),
                                           kernel_regularizer=regularizers.l2(l2_lambda)),
                              layers.Dropout(dropout),
                              layers.Dense(10,
                                           activation="softmax")])
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(
            x_val,
            y_val),
        epochs=50,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0)
    val_acc = max(history.history["val_accuracy"])
    filename = f"model_lr{
        lr:.5f}_units{units}_drop{
        dropout:.2f}_l2{
            l2_lambda:.5f}_batch{batch_size}.h5"
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    model.save(os.path.join("checkpoints", filename))
    return -val_acc


bounds = [{'name': 'learning_rate',
           'type': 'continuous',
           'domain': (1e-5,
                      1e-2)},
          {'name': 'num_units',
           'type': 'discrete',
           'domain': (64,
                      128,
                      256)},
          {'name': 'dropout_rate',
           'type': 'continuous',
           'domain': (0.0,
                      0.5)},
          {'name': 'l2_regularizer',
           'type': 'continuous',
           'domain': (1e-5,
                      1e-2)},
          {'name': 'batch_size',
           'type': 'discrete',
           'domain': (32,
                      64,
                      128)}]
# baseline validation accuracy before optimization
(x_train, y_train), (x_val, y_val) = load_dataset_mnist()
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax")
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop],
    verbose=0
)
val_acc = max(history.history["val_accuracy"])
print(f"Baseline validation accuracy: {val_acc:.4f}")
# optimization
opt = GPyOpt.methods.BayesianOptimization(
    f=train_model,
    domain=bounds,
    model_type='GP',
    acquisition_type='EI',
    maximize=False,
    initial_design_numdata=5,
    exact_feval=False)

opt.run_optimization(max_iter=30)
opt.plot_convergence()
with open('bayes_opt.txt', 'w') as f:
    f.write("best hyperparameters:\n")
    best_params = opt.x_opt
    f.write(f"learning rate: {best_params[0]:.5f}\n")
    f.write(f"number of units: {int(round(best_params[1]))}\n")
    f.write(f"dropout rate: {best_params[2]:.2f}\n")
    f.write(f"L2 regularizer: {best_params[3]:.5f}\n")
    f.write(f"batch size: {int(round(best_params[4]))}\n\n")
    f.write(f"best validation accuracy: {-opt.fx_opt:.4f}")

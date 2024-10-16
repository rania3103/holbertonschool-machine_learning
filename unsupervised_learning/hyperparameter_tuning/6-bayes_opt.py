#!/usr/bin/env python3
"""a python script that optimizes a machine learning
model of your choice using GPyOpt"""
import numpy as np
import GPy
import GPyOpt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# prepare a binary classification dataset
X, y = make_classification(
    n_samples=1000, n_features=20, n_classes=2, random_state=42)

# split the dataset into training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)
# set input dimension based on the feature size
input_dim = X_train.shape[1]
# function to create and compile a neural network model


def create_model(learning_rate, units, dropout_rate):
    model = Sequential()
    model.add(
        Dense(
            units=int(units),
            activation='relu',
            input_shape=(
                input_dim,
            )))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model
# function to evaluate the model with given hyperparameters


def evaluate_model(params):
    learning_rate = params[0][0]
    units = params[0][1]
    dropout_rate = params[0][2]
    batch_size = int(params[0][3])
    model = create_model(learning_rate, units, dropout_rate)
    # Early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    checkpoint = ModelCheckpoint(
        f'best_model_lr{learning_rate}_units{units}_dropout{dropout_rate}_batch{batch_size}.h5',
        save_best_only=True)
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50,
                        batch_size=batch_size,
                        callbacks=[early_stopping, checkpoint],
                        verbose=0)

    # return the validation loss as the metric to minimize
    return history.history['val_loss'][-1]


# define the search space for hyperparameters
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.001, 0.1)},
    {'name': 'units', 'type': 'discrete', 'domain': (16, 32, 64, 128)},
    {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.1, 0.5)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64)},
]
# initialize Bayesian Optimization
optimizer = GPyOpt.methods.BayesianOptimization(
    f=evaluate_model, domain=bounds)
# run optimization for 30 iterations
optimizer.run_optimization(max_iter=30)
# plot convergence
plt.plot(optimizer.Y)
plt.title("Bayesian Optimization Convergence")
plt.xlabel("Iteration")
plt.ylabel("Validation Loss")
plt.grid()
plt.savefig("convergence_plot.png")
plt.show()
# print the optimization results
print("Bayesian Optimization Report")
print("Best parameters:")
print(optimizer.x_opt)
print("Best validation loss:")
print(optimizer.Y.min())

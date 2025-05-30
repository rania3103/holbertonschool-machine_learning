#!/usr/bin/env python3
""" Transfer learning CIFAR-10 classifier using EfficientNetB0 from Keras Applications. """
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tqdm import tqdm
# disable GPU to avoid DNN init issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def preprocess_data(X, Y):
    X_p = X.astype('float32')
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = K.datasets.cifar10.load_data()
    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg"
    )
    base_model.trainable = False

    def compute_features(X, batch_size=256):
        features = []
        num_batches = int(np.ceil(len(X) / batch_size))
        for i in tqdm(
                range(num_batches),
                desc="Computing features",
                unit="batch"):
            batch_X = X[i * batch_size: (i + 1) * batch_size]
            batch_X_resized = tf.image.resize(batch_X, (224, 224))
            batch_X_pre = preprocess_input(batch_X_resized)
            batch_features = base_model.predict(batch_X_pre, verbose=0)
            features.append(batch_features)
        return np.vstack(features)
    print("computing features for training set")
    features_train = compute_features(X_train)
    print("training features done.")
    print("computing features for test set")
    features_test = compute_features(X_test)
    print("test features done.")
    head_inputs = K.Input(shape=features_train.shape[1:])
    x = layers.Dense(256, activation="relu")(head_inputs)
    x = layers.Dropout(0.5)(x)
    head_outputs = layers.Dense(10, activation="softmax")(x)
    head = K.Model(head_inputs, head_outputs, name="head")
    head.compile(
        optimizer=K.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    callbacks = [
        K.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        K.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6)
    ]
    head.fit(
        features_train, y_train,
        validation_data=(features_test, y_test),
        epochs=15,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )
    inputs = K.Input(shape=(32, 32, 3))
    x = layers.Resizing(224, 224)(inputs)
    x = layers.Lambda(preprocess_input, output_shape=(224, 224, 3))(x)
    x = base_model(x, training=False)
    outputs = head(x)
    model = K.Model(inputs, outputs, name="cifar10_efficientnetb0")
    model.compile(
        optimizer=K.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.save("cifar10.h5")
    print("model trained and saved to cifar10.h5")

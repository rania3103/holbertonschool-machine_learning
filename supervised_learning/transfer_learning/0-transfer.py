#!/usr/bin/env python3
""" a python script that trains a convolutional
neural network to classify the CIFAR 10 dataset"""
from tensorflow import keras as K


def preprocess_data(X, Y):
    """convert labels to a formar the model understands"""
    X_p = K.applications.resnet50.preprocess_input(X.astype('float32'))
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    """loads the cifar10 dataset, preprocesses the data,
    uses a pre trained resnet50 model to classify images
    and saves the trained model"""
    initializer = K.initializers.he_normal(seed=0)
    (X_train, y_train), (X_test, y_test) = K.datasets.cifar10.load_data()

    X_train_p, y_train_p = preprocess_data(X_train, y_train)
    X_test_p, y_test_p = preprocess_data(X_test, y_test)

    model = K.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3))

    for layer in model.layers:
        layer.trainable = False
    Model = K.Sequential([
        K.layers.Resizing(224, 224, interpolation='bilinear', input_shape=(32, 32, 3)),
        model,
        K.layers.Flatten(),
        K.layers.Dense(512, activation='relu', kernel_initializer=initializer),
        K.layers.Dropout(0.5),
        K.layers.Dense(10, activation='softmax', kernel_initializer=initializer)
    ])

    Model.compile(
        optimizer='adam',
        metrics=['accuracy'],
        loss='categorical_crossentropy')
    Model.fit(
        X_train_p,
        y_train_p,
        validation_data=(
            X_test_p,
            y_test_p),
        epochs=10,
        batch_size=128)
    Model.save('./cifar10.h5')

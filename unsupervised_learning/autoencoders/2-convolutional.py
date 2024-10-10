#!/usr/bin/env python3
"""a function that creates a convolutional autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Returns: encoder, decoder, auto"""
    input_lay = keras.Input(shape=input_dims)
    x = input_lay
    for filter in filters:
        x = keras.layers.Conv2D(
            filters=filter, kernel_size=(
                3, 3), padding='same', activation='relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    encoder_out = x
    encoder = keras.models.Model(input_lay, encoder_out, name='encoder')
    decoder_inp = keras.layers.Input(shape=encoder_out.shape[1:])
    x = decoder_inp
    for i, filter in enumerate(reversed(filters)):
        if i == len(filters) - 1:
            x = keras.layers.Conv2D(
                filters=filter, kernel_size=(
                    3, 3), padding='valid', activation='relu')(x)
        else:
            x = keras.layers.Conv2D(
                filters=filter, kernel_size=(
                    3, 3), padding='same', activation='relu')(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)
    x = keras.layers.Conv2D(
        filters=input_dims[-1], kernel_size=(3, 3),
        padding='same', activation='sigmoid')(x)
    decoder_output = x
    decoder = keras.models.Model(decoder_inp, decoder_output, name='decoder')
    autoencoder = keras.models.Model(
        inputs=input_lay,
        outputs=decoder(
            encoder(input_lay)),
        name='autoencoder')
    autoencoder.compile(optimizer='adam', loss="binary_crossentropy")
    return encoder, decoder, autoencoder

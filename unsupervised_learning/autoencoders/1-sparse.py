#!/usr/bin/env python3
"""a function that creates a sparse autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Returns: encoder, decoder, auto"""
    input_lay = keras.Input(shape=(input_dims,))
    x = input_lay
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha))(x)
    decoder_inp = latent
    for nodes in reversed(hidden_layers):
        decoder_inp = keras.layers.Dense(nodes, activation='relu')(decoder_inp)
    output = keras.layers.Dense(input_dims, activation='sigmoid')(decoder_inp)
    encoder_model = keras.Model(
        inputs=input_lay,
        outputs=latent,
        name='encoder')

    decoder_inp = keras.Input(shape=(latent_dims,))
    decoder_lay = decoder_inp
    for nodes in reversed(hidden_layers):
        decoder_lay = keras.layers.Dense(nodes, activation='relu')(decoder_lay)
    decoder_out = keras.layers.Dense(
        input_dims, activation='sigmoid')(decoder_lay)
    decoder_model = keras.Model(
        inputs=decoder_inp,
        outputs=decoder_out,
        name='decoder')

    autoencoder_inp = keras.Input(shape=(input_dims,))
    autoencoder_out = decoder_model(encoder_model(autoencoder_inp))
    autoencoder_model = keras.Model(
        inputs=autoencoder_inp,
        outputs=autoencoder_out,
        name='autoencoder')

    autoencoder_model.compile(optimizer='adam', loss="binary_crossentropy")
    return encoder_model, decoder_model, autoencoder_model

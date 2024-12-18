#!/usr/bin/env python3
"""Wasserstein GANs"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):
    """WGAN_clip class"""

    def __init__(
            self,
            generator,
            discriminator,
            latent_generator,
            real_examples,
            batch_size=200,
            disc_iter=2,
            learning_rate=.005):
        """constructor"""
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        # standard value, but can be changed if necessary
        self.beta_1 = .5
        # standard value, but can be changed if necessary
        self.beta_2 = .9

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: - \
            tf.math.reduce_mean(x)          # <----- new !
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(
            optimizer=generator.optimizer,
            loss=generator.loss)

        # define the discriminator loss and optimizer:
        self.discriminator.loss = lambda x, y: tf.math.reduce_mean(
            y) - tf.math.reduce_mean(x)  # <----- new !
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.compile(
            optimizer=discriminator.optimizer,
            loss=discriminator.loss)

    # generator of fake samples of size batch_size

    def get_fake_sample(self, size=None, training=False):
        """generate fake sample"""
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of real samples of size batch_size
    def get_real_sample(self, size=None):
        """generate real sample"""
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # overloading train_step()

    def train_step(self, useless_argument):
        """one training step"""
        for _ in range(self.disc_iter):

            # compute the loss for the discriminator in a tape watching the
            # discriminator's weights
            with tf.GradientTape() as tape:
                # get a real sample
                real_samples = self.get_real_sample()
                # get a fake sample
                fake_samples = self.get_fake_sample(training=True)
                # compute the loss discr_loss of the discriminator on real and
                # fake samples
                discr_loss_real = self.discriminator(
                    real_samples, training=True)
                discr_loss_fake = self.discriminator(
                    fake_samples, training=True)
                discr_loss = self.discriminator.loss(
                    discr_loss_real, discr_loss_fake)
            # apply gradient descent once to the discriminator
            discr_grad = tape.gradient(
                discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(discr_grad, self.discriminator.trainable_variables))
            # clip the weights (of the discriminator) between -1 and 1    #
            # <----- new !
            for v in self.discriminator.trainable_variables:
                v.assign(tf.clip_by_value(v, -1.0, 1.0))
        # compute the loss for the generator in a tape watching the generator's
        # weights
        with tf.GradientTape() as tape:
            # get a fake sample
            fake_samples = self.get_fake_sample(training=True)
            fake_output = self.discriminator(fake_samples, training=True)
            # compute the loss gen_loss of the generator on this sample
            gen_loss = self.generator.loss(fake_output)
        # apply gradient descent to the generator
        discr_grad_desc = tape.gradient(
            gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(discr_grad_desc, self.generator.trainable_variables))
        return {"discr_loss": discr_loss, "gen_loss": gen_loss}

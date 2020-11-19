#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:23:28 2020

@author: prashantcraju
"""

class VAE:

    def __init__(self, input_shape, latent_dim, beta):
        self.input_shape=input_shape
        self.latent_dim=latent_dim
        self.beta=beta
        self.od=self.outlier_vae()
#        self.thre=threshold
    
    def encoder_net(self):
        import tensorflow as tf
        from tensorflow.keras.layers import Conv2D, InputLayer
        return tf.keras.Sequential([
          InputLayer(input_shape=self.input_shape),
          Conv2D(self.input_shape[0]+self.input_shape[1], 4, strides=2, padding='same', activation=tf.nn.relu),
          Conv2D(((self.input_shape[0]+self.input_shape[1])*2), 4, strides=2, padding='same', activation=tf.nn.relu),
          Conv2D(((self.input_shape[0]+self.input_shape[1])*8), 4, strides=2, padding='same', activation=tf.nn.relu)])

    def decoder_net(self):
        import tensorflow as tf
        from tensorflow.keras.layers import Conv2DTranspose, Dense, Reshape, InputLayer
        return tf.keras.Sequential([
          InputLayer(input_shape=(self.latent_dim,)),
          Dense(4*4*((self.input_shape[0]+self.input_shape[1])*2)),
          Reshape(target_shape=(4, 4, ((self.input_shape[0]+self.input_shape[1])*2))),
          Conv2DTranspose(((self.input_shape[0]+self.input_shape[1])*4), 4, strides=2, padding='same', activation=tf.nn.relu),
          Conv2DTranspose((self.input_shape[0]+self.input_shape[1]), 4, strides=2, padding='same', activation=tf.nn.relu),
          Conv2DTranspose(self.input_shape[2], 4, strides=2, padding='same', activation='sigmoid')])
    
    def outlier_vae(self):
        from alibi_detect.od import OutlierVAE
        return OutlierVAE(threshold=0.15,
                    score_type='mse', 
                    encoder_net=self.encoder_net(),  
                    decoder_net=self.decoder_net(),
                    latent_dim=self.latent_dim,
                    beta=self.beta,
                    samples=2)
    
    def fit(self, data, epochs, verbose, filepath):
        from alibi_detect.models.losses import elbo
        from alibi_detect.utils.saving import save_detector

        self.od.fit(data, loss_fn=elbo, epochs=epochs,verbose=verbose)
        save_detector(self.od, filepath)

    def load_model(self, filepath):
        from alibi_detect.utils.saving import load_detector
        self.od = load_detector(filepath)
        
    def predict(self, data):
        return self.od.predict(data,
                      outlier_type='instance',    
                      return_feature_score=True,
                      return_instance_score=True)
        
        
        
        
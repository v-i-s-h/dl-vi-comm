# AE Comm System Oshea

# VAE Comm Module

from tensorflow.keras.layers import Dense, Input, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import backend as K
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist # For calculating QPSK decoding

import datetime, itertools, dill

class AEOshea1hot(object):
  def __init__(self, in_dim=None, latent_dim=None, h_dim=None, train_snr_dB=10.0):
    self.in_dim = in_dim
    self.latent_dim = latent_dim
    self.train_snr_dB = train_snr_dB
    self.h_dim = h_dim
    self.train_noisepow = 10.0**(-self.train_snr_dB/10.0)
    
    if self.in_dim and self.latent_dim:
      self.make_model()
    
  def make_model(self):
    # Input layer
    self.inputs = Input(shape=(self.in_dim,), name="enc_in")
    
    # Hidden Layers
    x = self.inputs
    if self.h_dim is not None:
      for (i,d) in enumerate(self.h_dim):
        x = Dense( d, activation='relu', name="enc_l{}".format(i))(x)
    # Mean and Variance
    x = Dense(self.latent_dim)(x)
    self.z_mean = BatchNormalization(center=False, scale=False)(x)
    # Channel
    self.z = Lambda(self.channel, output_shape=(self.latent_dim,), name="z")(self.z_mean)
   
    
    # Encoder model
    self.encoder = Model(self.inputs, [self.z_mean,self.z], name="encoder")
    
    # Decoder
    self.latent_inputs = Input(shape=(self.latent_dim,), name="z_sample")
    
    # Hidden layers
    x = self.latent_inputs
    if self.h_dim is not None:
      for (i,d) in enumerate(self.h_dim[::-1]):
        x = Dense( d, activation='relu', name="dec_l{}".format(i))(x)
    self.dec_outputs = Dense( self.in_dim, activation='softmax', name="decoder_out")(x)
    
    # Decoder model
    self.decoder = Model(self.latent_inputs, self.dec_outputs, name="decoder")
    
    # VAE
    self.outputs = self.decoder(self.encoder(self.inputs)[1])
    self.model = Model(self.inputs, self.outputs, name="VAE")
    
    # Losses
    self.recon_loss = categorical_crossentropy(self.inputs, self.outputs)
    
    self.model.add_loss(self.recon_loss)
    
    self.model.compile(optimizer='adam')
#     self.model.compile( optimizer=tf.train.AdamOptimizer(learning_rate=0.01))
    
    
  def channel( self, zMean ):
    batch = K.shape( zMean )[0]
    epsilon = K.random_normal( shape = (batch,self.latent_dim) )
    # Because BatchNormalization produces z vector with signal power 'latentDim',
    # we should not scale the noise power here.
    return zMean + np.sqrt(self.train_noisepow)*epsilon
  
  def fit(self, x_train, epochs=10, batch_size=128, validation_data=None, 
          verbose=0, callbacks=None):
    train_log = self.model.fit(x_train, epochs=epochs, batch_size=batch_size, 
                               validation_data=validation_data, verbose=verbose,
                               callbacks=callbacks)
    return train_log.history
  
  def encode( self, data ):
    return self.encoder.predict(data)
  
  def decode( self, data ):
    return self.decoder.predict(data)
  
  def analysis( self ):
    xTest = np.eye(self.in_dim)
    enc_mu, enc_z = self.encode(xTest)
    dec_mu = self.decode(enc_mu)
    dec_z = self.decode(enc_z)

    chDim = self.latent_dim//2
    f = plt.figure(figsize=(5*chDim,9))
    for i in range(chDim):
      ax1 = plt.subplot(2,chDim,i+1)
      ax1.scatter(enc_mu[:,i],enc_mu[:,i+chDim],c=np.arange(self.in_dim))
      for j in range(self.in_dim):
        ax1.annotate( j, (enc_mu[j,i],enc_mu[j,i+chDim]) )
      ax1.set_title( "TX Symbols n = {}".format(i+1) )

      
  def save_model(self, fileprefix):
    with open(fileprefix+".dil","wb") as obj:
      dill.dump( {'in_dim': self.in_dim,
                  'latent_dim': self.latent_dim,
                  'train_noisepow': self.train_noisepow,
                  'train_snr_dB': self.train_snr_dB,
                  'h_dim': self.h_dim}, obj )
    self.model.save_weights(fileprefix+".h5")
  
  def load_model(self, fileprefix):
    with open(fileprefix+".dil","rb") as obj:
      config = dill.load(obj)
      self.in_dim = config['in_dim']
      self.latent_dim = config['latent_dim']
      self.train_snr_dB = config['train_snr_dB']
      self.train_noisepow = config['train_noisepow']
      self.h_dim = config['h_dim']
    self.make_model()
    self.model.load_weights(fileprefix+".h5")
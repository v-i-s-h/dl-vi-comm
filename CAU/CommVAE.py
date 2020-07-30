# VAE Comm Module

from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.contrib.distributions import Cauchy
from tensorflow.keras import backend as K
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist # For calculating QPSK decoding

import datetime, itertools, dill

class CommVAE1hot(object):
  """
    in_dim : block length
    latent_dim : encoding dimension (half of number of channel uses)
    h_dim : number of hidden layers
    obj_fn : objective function with to optimize over
    n0 : noise power (over all components)
    sigma2 : prior variance (per component)
  """
  def __init__(self, in_dim=None, latent_dim=None, h_dim=None, obj_fn='RBF', n0=1.0, sigma2=1.0):
    self.in_dim = in_dim
    self.latent_dim = latent_dim
    self.n0 = n0
    self.sigma2 = sigma2
    self.h_dim = h_dim
    self.obj_fn = obj_fn    
    
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
    self.z_mean = Dense(self.latent_dim, name="z_mean")(x)
    
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

    if self.obj_fn == 'AWGN':
      # print( "Model with AWGN ")
      sig_pow = 0.5 * 1.0/self.sigma2 * K.sum(K.square(self.z_mean), axis=-1) 
      noise_term = 0.5 * self.latent_dim * ((self.n0/self.latent_dim)/self.sigma2 - 
                                        1.0 - K.log((self.n0/self.latent_dim)/self.sigma2))
      self.kl_loss = sig_pow + noise_term
    elif self.obj_fn == 'RBF':
      # print( "Model with RBF")
      sig_pow = 0.5 * 1.0/self.sigma2 * K.sum(K.square(self.z_mean), axis=-1) 
      noise_term = 0.5 * self.latent_dim * ((self.n0/self.latent_dim)/self.sigma2 - 
                                        1.0 - K.log((self.n0/self.latent_dim)/self.sigma2))
      rbf_term = K.log(1.0 + 0.5*self.latent_dim/self.n0 * sig_pow)
      self.kl_loss = sig_pow + noise_term - rbf_term
    elif self.obj_fn == "CAU":
        # From: https://arxiv.org/abs/1905.10965
        n0 = self.n0 / self.latent_dim
        self.kl_loss = K.sum(K.log( ((n0+self.sigma2)**2 + K.square(self.z_mean)) / (4*n0*self.sigma2) ), axis=-1)
    else:
      raise NotImplementedError("Unknown obj_fn: {}".format(self.obj_fn))
    
    
    self.vae_loss = K.mean( self.recon_loss + self.kl_loss )

    self.model.add_loss( self.vae_loss )
    
    self.model.compile( optimizer = 'adam' )
#     self.model.compile( optimizer=tf.train.AdamOptimizer(learning_rate=0.01))
    
    
  def channel( self, zMean ):
    batch = K.shape( zMean )[0]
#     dims = K.shape( zMean )[1]
    noise_distrib = Cauchy(np.zeros(self.latent_dim), np.ones(self.latent_dim))
    epsilon = tf.cast(noise_distrib.sample( sample_shape = (batch) ), tf.float32)
    
    # return zMean + np.sqrt(self.n0/self.latent_dim)*epsilon
    return zMean + self.n0/self.latent_dim * epsilon
  
  # def fit(self, x_train, epochs=10, batch_size=128, validation_data=None, verbose=0 ):
  #   train_log = self.model.fit( x_train, epochs=epochs, batch_size=batch_size, 
  #                               validation_data=validation_data, verbose=verbose )
  #   return train_log.history
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
                  'n0': self.n0,
                  'sigma2': self.sigma2,
                  'h_dim': self.h_dim,
                  'obj_fn': self.obj_fn}, obj )
    self.model.save_weights(fileprefix+".h5")
  
  def load_model(self, fileprefix):
    with open(fileprefix+".dil","rb") as obj:
      config = dill.load(obj)
      self.in_dim = config['in_dim']
      self.latent_dim = config['latent_dim']
      self.n0 = config['n0']
      self.h_dim = config['h_dim']
      self.obj_fn = config['obj_fn']
      if 'sigma2' in config:  # For backward compatability
        self.sigma2 = config['sigma2']
      else:
        self.sigma2 = 1.0
    self.make_model()
    self.model.load_weights(fileprefix+".h5")
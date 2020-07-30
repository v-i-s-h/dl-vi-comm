from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.layers import Conv2D, Flatten, Lambda
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy, categorical_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json

import numpy as np
import itertools
import dill

class CommVAEBinary(object):
  def __init__(self, in_dim=None, latent_dim=None, h_dim=None, obj_fn = 'RBF', n0=1.0, sigma0_2=1.0):
    self.in_dim = in_dim
    self.latent_dim = latent_dim
    self.n0 = n0
    self.h_dim = h_dim
    self.obj_fn = obj_fn
    self.sigma0_2 = sigma0_2
    
    
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
    self.dec_outputs = Dense( self.in_dim, activation='sigmoid', name="decoder_out")(x)
    
    # Decoder model
    self.decoder = Model(self.latent_inputs, self.dec_outputs, name="decoder")
    
    # VAE
    self.outputs = self.decoder(self.encoder(self.inputs)[1])
    self.model = Model(self.inputs, self.outputs, name="VAE")
    
    # Losses
    self.recon_loss = binary_crossentropy(self.inputs, self.outputs)
    self.recon_loss *= self.in_dim # Because binary_crossentropy divides by N


    if self.obj_fn == 'AWGN':
      # print( "Model with AWGN ")
      sig_pow = 0.5/self.sigma0_2 * K.sum(K.square(self.z_mean), axis=-1) 
      noise_term = 0.5 * self.latent_dim * (self.n0/self.latent_dim/self.sigma0_2 - 1.0 - K.log(self.n0/self.latent_dim/self.sigma0_2))
      self.kl_loss = sig_pow + noise_term
    elif self.obj_fn == 'RBF':
      # print( "Model with RBF")
      sig_pow = 0.5/self.sigma0_2 * K.sum(K.square(self.z_mean), axis=-1) 
      noise_term = 0.5 * self.latent_dim * (self.n0/self.latent_dim/self.sigma0_2 - 1.0 - K.log(self.n0/self.latent_dim/self.sigma0_2))
      rbf_term = K.log(1.0 + 0.5*self.latent_dim/self.n0 * sig_pow)
      self.kl_loss = sig_pow + noise_term - rbf_term
    else:
      raise NotImplementedError("Unknown obj_fn: {}".format(self.obj_fn))
    
    
    self.vae_loss = K.mean( self.recon_loss + self.kl_loss )

    self.model.add_loss( self.vae_loss )
    
    self.model.compile( optimizer = 'adam' )
#     self.model.compile( optimizer=tf.train.AdamOptimizer(learning_rate=0.01))
    
    
  def channel( self, zMean ):
    batch = K.shape( zMean )[0]
#     dims = K.shape( zMean )[1]
    epsilon = K.random_normal( shape = (batch,self.latent_dim) )
    return zMean + np.sqrt(self.n0/self.latent_dim)*epsilon
  
  def fit(self, x_train, epochs=10, batch_size=128, validation_data=None, verbose=0 ):
    train_log = self.model.fit( x_train, epochs=epochs, batch_size=batch_size, 
                                validation_data=validation_data, verbose=verbose )
    return train_log.history
  
  def encode( self, data ):
    return self.encoder.predict(data)
  
  def decode( self, data ):
    return self.decoder.predict(data)
  
#   def analysis( self ):
#     xTest = np.eye(self.in_dim)
#     enc_mu, enc_z = self.encode(xTest)
#     dec_mu = self.decode(enc_mu)
#     dec_z = self.decode(enc_z)

#     chDim = self.latent_dim//2
#     f = plt.figure(figsize=(5*chDim,9))
#     for i in range(chDim):
#       ax1 = plt.subplot(2,chDim,i+1)
#       ax1.scatter(enc_mu[:,i],enc_mu[:,i+chDim],c=np.arange(self.in_dim))
#       for j in range(self.in_dim):
#         ax1.annotate( j, (enc_mu[j,i],enc_mu[j,i+chDim]) )
#       ax1.set_title( "TX Symbols n = {}".format(i+1) )
      
  def analysis( self ):    
    xTest = np.array(list(map(list, itertools.product([0,1], repeat=self.in_dim))))
    enc_mu, enc_z = self.encode(xTest)
    dec_mu = self.decode(enc_mu)
    dec_z = self.decode(enc_z)

    chDim = self.latent_dim//2
    f = plt.figure(figsize=(5*chDim,9))
    for i in range(chDim):
      ax1 = plt.subplot(2,chDim,i+1)
      ax1.scatter(enc_mu[:,i],enc_mu[:,i+chDim],c=np.arange(2**self.in_dim))
      for j in range(2**self.in_dim):
        ax1.annotate( j, (enc_mu[j,i],enc_mu[j,i+chDim]) )
      ax1.set_title( "TX Symbols n = {}".format(i+1) )
  
  def get_constellation(self):
    xTest = np.array(list(map(list, itertools.product([0,1], repeat=self.in_dim))))
    enc_mu, enc_z = self.encode(xTest)
    
    return enc_mu
      
  def save_model(self, fileprefix):
    with open(fileprefix+".dil","wb") as obj:
      dill.dump( {'in_dim': self.in_dim,
                  'latent_dim': self.latent_dim,
                  'n0': self.n0,
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
    self.make_model()
    self.model.load_weights(fileprefix+".h5")
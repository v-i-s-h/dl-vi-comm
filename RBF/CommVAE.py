# VAE Comm Module

from tensorflow.keras.layers import Dense, Input, Lambda, Concatenate, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import backend as K
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist # For calculating QPSK decoding

import datetime, itertools, dill

class CommVAE1hot(object):
  def __init__(self, in_dim=None, latent_dim=None, h_dim=None, obj_fn='RBF', n0=1.0):
    self.in_dim = in_dim
    self.latent_dim = latent_dim
    self.n0 = n0
    self.h_dim = h_dim
    self.obj_fn = obj_fn
    
    self.pilot_sym = [+1.0, +1.0]

    
    
    if self.in_dim and self.latent_dim:
      self.make_model()
    
  def make_model(self):
    # ------------------------------ ENCODER ---------------------------
    # Input layer
    self.enc_in = Input(shape=(self.in_dim,), name="enc_in")
    # Hidden Layers
    x = self.enc_in
    if self.h_dim is not None:
      for (i,d) in enumerate(self.h_dim):
        x = Dense( d, activation='relu', name="enc_l{}".format(i))(x)
    # Mean
    self.z_mean = Dense(self.latent_dim, name="z_mean")(x)
    # Encoder model
    self.encoder = Model(self.enc_in, self.z_mean, name="encoder")
    
    # ----------------------------- PRE-TX MODEL ----------------------
    self.pretx_in = Input(shape=(self.latent_dim,), name="pretx_in")
    self.tx_blk = Lambda(self.add_pilot, output_shape=(self.latent_dim+2,), 
                    name="tx_blk")(self.pretx_in) 
    self.pretx_model = Model(self.pretx_in, self.tx_blk, name="pretx_model")
    
    # ----------------------------- FADING MODEL ----------------------
    self.fading_in = Input(shape=(self.latent_dim+2,), name="fading_in")
    self.faded_blk = Lambda(self.rayleigh_block_fading, output_shape=(self.latent_dim+2,), 
                            name="faded_blk")(self.fading_in)
    self.fading_model = Model(self.fading_in, self.faded_blk, name="fading_model")
    
    # --------------------------- PRE NOISE MODEL -------------------
    self.prenoise_out = self.fading_model(self.pretx_model(self.encoder(self.enc_in)))
    self.prenoise_model = Model(self.enc_in, self.prenoise_out, name="prech_model")
    
    # ----------------------------- NOISE MODEL -----------------------
    self.ch_in = Input(shape=(self.latent_dim+2,), name="ch_in")
    self.ch_out = Lambda(self.channel, output_shape=(self.latent_dim+2,), 
                         name="ch_out")(self.ch_in)
    self.noise_model = Model(self.ch_in, self.ch_out, name="noise_model")
    
    # ----------------------------- PRE-DEC MODEL ---------------------
    self.predec_in = Input(shape=(self.latent_dim+2,), name="predec_in")
    self.predec_out = Lambda(self.equalizer, output_shape=(self.latent_dim,), 
                             name="predec_out")(self.predec_in)
    self.predec_model = Model(self.predec_in, self.predec_out, name="predec_model")
    
    # ----------------------------- DECODER ---------------------------
    self.latent_inputs = Input(shape=(self.latent_dim,), name="latent_inputs")
    # Hidden layers
    x = self.latent_inputs
    if self.h_dim is not None:
      for (i,d) in enumerate(self.h_dim[::-1]):
        x = Dense( d, activation='relu', name="dec_l{}".format(i))(x)
    self.dec_out = Dense( self.in_dim, activation='softmax', name="decoder_out")(x)
    # Decoder model
    self.decoder = Model(self.latent_inputs, self.dec_out, name="decoder")
    
    # -------------------------- POST NOISE MODEL ---------------------
    self.postnoise_dec_out = self.decoder(self.predec_model(self.predec_in))
    self.postnoise_model = Model(self.predec_in, self.postnoise_dec_out, 
                                 name="postnoise_dec_out")
    
    
    # ---------------------------- VAE MODEL --------------------------
#     self.outputs = self.decoder(
#                       self.predec_model(
#                           self.noise_model(
#                               self.fading_model(
#                                   self.pretx_model(
#                                       self.encoder(
#                                           self.enc_in
#                                       )
#                                   )
#                               )
#                           )
#                       )
#                    )
    self.outputs = self.postnoise_model(
                     self.noise_model(
                         self.prenoise_model(
                             self.enc_in
                         )
                     )  
                  )
    # VAE
    self.model = Model(self.enc_in, self.outputs, name="VAE")
    
    
    # ---------------------------- LOSSES -----------------------------
    self.recon_loss = categorical_crossentropy(self.enc_in, self.outputs)

    if self.obj_fn == 'AWGN':
      # print( "Model with AWGN ")
      sig_pow = 0.5 * K.sum(K.square(self.z_mean), axis=-1) 
      noise_term = 0.5 * self.latent_dim * (self.n0/self.latent_dim - 1.0 - K.log(self.n0/self.latent_dim))
      self.kl_loss = sig_pow + noise_term
    elif self.obj_fn == 'RBF':
      # print( "Model with RBF")
      sig_pow = 0.5 * K.sum(K.square(self.z_mean), axis=-1) 
      noise_term = 0.5 * self.latent_dim * (self.n0/self.latent_dim - 1.0 - K.log(self.n0/self.latent_dim))
      rbf_term = K.log(1.0 + 0.5*self.latent_dim/self.n0 * sig_pow)
      self.kl_loss = sig_pow + noise_term - rbf_term
    else:
      raise NotImplementedError("Unknown obj_fn: {}".format(self.obj_fn))
    
    
    self.vae_loss = K.mean( self.recon_loss + self.kl_loss )

    self.model.add_loss( self.vae_loss )
    
    self.model.compile( optimizer = 'adam' )
#     self.model.compile( optimizer=tf.train.AdamOptimizer(learning_rate=0.01))
    
    
  def channel( self, faded_blk ):
    # Add AWGN noise
    epsilon = K.random_normal(shape=K.shape(faded_blk))
    
    # Each component is affected by n0/latent_dim, even pilots.
    return faded_blk + np.sqrt(self.n0/self.latent_dim)*epsilon
  
  
  def add_pilot(self, sym_blk):

    batch_size = K.shape(sym_blk)[0]
    enc_dim = K.shape(sym_blk)[1]

    # Pack Pilot1 + Real Part + Pilot2 + Imag Part
    tx_blk = K.concatenate( [self.pilot_sym[0]*tf.ones((batch_size,1)),
                             sym_blk[:,:enc_dim//2],
                             self.pilot_sym[1]*tf.ones((batch_size,1)),
                             sym_blk[:,enc_dim//2:]], axis=1 )

    return tx_blk
  
  def rayleigh_block_fading(self, symbol_tensor):
  
    batch_size = K.shape(symbol_tensor)[0]
    enc_dim = K.shape(symbol_tensor)[1]

    h1_tensor = K.random_normal(shape=(batch_size,), stddev=np.sqrt(0.5))
    h2_tensor = K.random_normal(shape=(batch_size,), stddev=np.sqrt(0.5))

    p2 = K.concatenate( [-symbol_tensor[:,enc_dim//2:],symbol_tensor[:,:enc_dim//2]] )

    t1 = h1_tensor[:,None]*symbol_tensor + h2_tensor[:,None]*p2

    return t1
  
  def equalizer(self, rx_blk):
  
    batch_size = K.shape(rx_blk)[0]
    enc_dim = K.shape(rx_blk)[1]


    rx_pilot0 = rx_blk[:,0]
    rx_pilot1 = rx_blk[:,enc_dim//2]

    h1_hat = (self.pilot_sym[1]*rx_pilot1+self.pilot_sym[0]*rx_pilot0)/(self.pilot_sym[0]**2+self.pilot_sym[1]**2)
    h2_hat = (self.pilot_sym[0]*rx_pilot1-self.pilot_sym[1]*rx_pilot0)/(self.pilot_sym[0]**2+self.pilot_sym[1]**2)    

    z1_hat = rx_blk[:,:enc_dim//2]
    z2_hat = rx_blk[:,enc_dim//2:]


    zR = (h1_hat[:,None]*z1_hat+h2_hat[:,None]*z2_hat) / (h1_hat[:,None]**2+h2_hat[:,None]**2)
    zI = (h1_hat[:,None]*z2_hat-h2_hat[:,None]*z1_hat) / (h1_hat[:,None]**2+h2_hat[:,None]**2)

    # First in each split is pilot, remove them
    out_blk = K.concatenate((zR[:,1:],zI[:,1:]))
  
    return out_blk

  
#   def fit(self, x_train, epochs=10, batch_size=128, validation_data=None, verbose=0 ):
# #     train_n0 = self.n0 * np.ones((x_train.shape[0],1))
# #     if validation_data is not None:
# #       xVal, yVal = validation_data
# #       val_n0 = self.n0 * np.ones((xVal.shape[0],1))
# #       validation_data = ([xVal,val_n0], yVal)
#     train_log = self.model.fit( x_train, epochs=epochs, batch_size=batch_size, 
#                                 validation_data=validation_data, verbose=verbose )
#     return train_log.history

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
  
  def prenoise_encode(self, data):
    return self.prenoise_model.predict(data)
  
  def postnoise_decode(self, data):
    return self.postnoise_model.predict(data)
  
  def analysis( self ):
    xTest = np.eye(self.in_dim)
    enc_mu  = self.encode(xTest)
    dec_mu = self.decode(enc_mu)
    
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
                  'h_dim': self.h_dim,
                  'obj_fn': self.obj_fn,
                  'pilot_sym': self.pilot_sym}, obj )
    self.model.save_weights(fileprefix+".h5")
  
  def load_model(self, fileprefix):
    with open(fileprefix+".dil","rb") as obj:
      config = dill.load(obj)
      self.in_dim = config['in_dim']
      self.latent_dim = config['latent_dim']
      self.n0 = config['n0']
      self.h_dim = config['h_dim']
      self.obj_fn = config['obj_fn']
      self.pilot_sym = config['pilot_sym']
    self.make_model()
    self.model.load_weights(fileprefix+".h5")
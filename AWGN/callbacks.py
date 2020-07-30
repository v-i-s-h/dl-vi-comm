# Keras Callbacks for monitoring training progress

import numpy as np
from scipy.spatial.distance import cdist # For calculating distances
import tensorflow as tf


class PackingDensityMonitor(tf.keras.callbacks.Callback):
    def __init__(self, model, in_dim, interval=100):
        super().__init__()
        self.dl_model = model
        self.in_dim = in_dim
        self.interval = interval
        self.results = {
            "epochs": [],
            "en": [],
            "dmin": []
        }

        # Create codebook
        self.one_hot_code = np.eye(self.in_dim)

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        """
            Evaluate packing density at preset intevals
        """
        if epoch % self.interval == 0:
            dmin, en = self.compute_metrics()
            self.results["epochs"].append(epoch)
            self.results["en"].append(en)
            self.results["dmin"].append(dmin)

    def on_train_begin(self, logs=None):
        """
        """
        self.values = []    # Clear the list at the event of starting

    def on_train_end(self, logs=None):
        dmin, en = self.compute_metrics()
        # Crude hack to find last epoch index
        self.results["epochs"].append(self.results["epochs"][-1] + self.interval)   
        self.results["en"].append(en)
        self.results["dmin"].append(dmin)

    def compute_metrics(self):
        # Compute the Tx power and packing density
        dl_map, _ = self.dl_model.encode(self.one_hot_code)
        dl_sym_pow = np.mean(np.sum(dl_map*dl_map,axis=1))
        unique_sym_distances = np.unique(cdist(dl_map,dl_map))
        if len(unique_sym_distances) == 1: # All distances are same and will be zero
            dl_d_min = np.inf  # This is not a valid point
            dl_en = np.nan
        else:
            dl_d_min = np.unique(cdist(dl_map,dl_map))[1]
            dl_en = dl_sym_pow / (dl_d_min**2)

        return dl_d_min, dl_en
        

class BlerMonitor(tf.keras.callbacks.Callback):
    def __init__(self, model, in_dim, ch_use, snr_dB, interval=100):
        super().__init__()
        self.dl_model = model
        self.in_dim = in_dim
        self.snr_dB = snr_dB
        self.ch_use = ch_use
        self.interval = interval
        self.results = {
            "epochs": [],
            "bler": []
        }

        # Create codebook
        self.one_hot_code = np.eye(self.in_dim)

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        """
            Evaluate packing density at preset intevals
        """
        if epoch % self.interval == 0:
            bler = self.compute_metrics()
            self.results["epochs"].append(epoch)
            self.results["bler"].append(bler)

    def on_train_begin(self, logs=None):
        """
        """
        self.values = []    # Clear the list at the event of starting

    def on_train_end(self, logs=None):
        bler = self.compute_metrics()
        # Crude hack to find last epoch index
        self.results["epochs"].append(self.results["epochs"][-1] + self.interval)   
        self.results["bler"].append(bler)

    def compute_metrics(self):
        z_mu, _ = self.dl_model.encode(np.eye(self.in_dim))
        sym_pow = np.mean(np.sum(z_mu*z_mu,axis=1))
        noisePower = sym_pow * 10.0**(-self.snr_dB/10.0)
        n0 = noisePower / (2*self.ch_use)

        thisErr = 0
        thisCount = 0
        while thisErr < 500:
            txSym = np.random.randint(self.in_dim, size=1000)
            tx1hot = np.eye(self.in_dim)[txSym]
            txTest, _ = self.dl_model.encode(tx1hot)
            rxTest = txTest + np.random.normal(scale=np.sqrt(n0), size=txTest.shape)
            rxDecode = self.dl_model.decode(rxTest)
            rxSym = np.argmax(rxDecode,axis=1)
            thisErr += np.sum(rxSym!=txSym)
            thisCount += 1000
        blkErr = thisErr / thisCount

        return blkErr
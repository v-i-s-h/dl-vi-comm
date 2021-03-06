# Usage: python main.py --blksize 8 --chuse 4 --objfn RBF --n0 0.8 --epochs 2500 --models 100 --prefix ./models/awgn_rbf_64_32_16_n080
import numpy as np
import datetime, os, sys, argparse, time
import dill 
import tensorflow.keras.backend as K

from CommVAE import CommVAE1hot

import callbacks

parser  = argparse.ArgumentParser()
parser.add_argument('--blksize', type=int, help="Block Size", default=8)
parser.add_argument('--chuse', type=int, help="No. of channel uses", default=4 )
parser.add_argument('--objfn', type=str, help="Cost function [AWGN, RBF]", default="RBF" )
parser.add_argument('--n0', type=float, help="Noise power (combined)", default=0.8 )
parser.add_argument('--sigma2', type=float, help="Prior variance", default=1.0)
parser.add_argument('--epochs', type=int, help='No.of epochs to train', default=2500 )
parser.add_argument('--models', type=int, help='No.of models to train', default=100 )
parser.add_argument('--prefix', type=str, help='Output directory', default = './test/' )
args = parser.parse_args()

noOfModels  = args.models
noOfEpochs  = args.epochs
blkSize     = args.blksize
chUse       = args.chuse
outPrefix   = args.prefix
objFn       = args.objfn
trainN0     = args.n0
sigma2      = args.sigma2

SNR_range_dB = np.arange( 0.0, 11.0, 1.0 )

# Input
inVecDim   = 2 ** blkSize  # 1-hot vector length for block
encDim = 2*chUse
print( "In Vector Dim:", inVecDim )
print( "z dim:", encDim )

# Train Data
xTrain = np.eye(inVecDim)[np.random.randint(inVecDim,size=10000)]
xVal = np.eye(inVecDim)[np.random.randint(inVecDim,size=10000)]

# ==========================================================================================
results = {}
for i in range(noOfModels):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    print("{} Training {:3d}/{}".format(timestamp,i+1,noOfModels))
    
    # Clear Model
    K.clear_session()
    try:
        del m
    except:
        pass

    m = CommVAE1hot(inVecDim, encDim, h_dim=[64,32,16], 
                        n0=trainN0, sigma2=sigma2, obj_fn=objFn)
    
    # Create callback monitor
    monitors = {
        "packing_density": callbacks.PackingDensityMonitor(m, 
                                                           in_dim=inVecDim, 
                                                           interval=25), # Packing density monitor
        "bler_4dB": callbacks.BlerMonitor(m, in_dim=inVecDim, 
                                          ch_use=chUse, snr_dB=4.0, 
                                          interval=100),
        "bler_6dB": callbacks.BlerMonitor(m, in_dim=inVecDim, 
                                          ch_use=chUse, snr_dB=6.0, 
                                          interval=100)
    }

    h = m.fit(xTrain, epochs=noOfEpochs, batch_size=1000, verbose=0, 
                callbacks=[ cb for (_, cb) in monitors.items() ])
    m.save_model(outPrefix + "_" + timestamp)

    z_mu, _ = m.encode(np.eye(inVecDim))
    sym_pow = np.mean(np.sum(z_mu*z_mu,axis=1))
    print( "Avg. Tx Sym Power:", sym_pow )
    noisePower = sym_pow * 10.0**(-SNR_range_dB/10.0)
    n0_per_comp = noisePower/(2*chUse)

    err = []
    for n0 in n0_per_comp:
        thisErr = 0
        thisCount = 0
        while thisErr < 500:
            txSym = txSym = np.random.randint(inVecDim, size=1000)
            tx1hot = np.eye(inVecDim)[txSym]
            txTest, _ = m.encode(tx1hot)
            rxTest = txTest + np.random.normal(scale=np.sqrt(n0), size=txTest.shape)
            rxDecode = m.decode(rxTest)
            rxSym = np.argmax(rxDecode,axis=1)
            thisErr += np.sum(rxSym!=txSym)
            thisCount += 1000
        err.append(thisErr/thisCount)
    blkErr = np.array(err)
    
    # Make results entry
    results[timestamp] = {
        "sym_pow": sym_pow, 
        "snr_dB": SNR_range_dB,
        "bler": blkErr
    }
    # Merge all monitor results
    for label, monitor in monitors.items():
        for (k, v) in monitor.results.items():
            results[timestamp]["{}_{}".format(label, k)] = v

with open(outPrefix+"_summary.dil", "wb") as f:
    dill.dump(results, f)
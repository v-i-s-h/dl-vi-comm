# Usage: python main.py --blksize 8 --chuse 4 --objfn RBF --n0 0.8 --epochs 3000 --models 100 --prefix ./models/rbf_rbf_64_32_16_n080

import numpy as np
import datetime, os, sys, argparse, time
import dill 
import tensorflow.keras.backend as K

from CommVAE_Aoudia import CommVAE1hot_Aoudia

parser  = argparse.ArgumentParser()
parser.add_argument('--blksize', type=int, help="Block Size", default=8)
parser.add_argument('--chuse', type=int, help="No. of channel uses", default=4 )
parser.add_argument('--objfn', type=str, help="Cost function [AWGN, RBF]", default="RBF" )
parser.add_argument('--n0', type=float, help="Noise power (combined)", default=0.8 )
parser.add_argument('--sigma', type=float, help="Sigma for TX output perturbation", default=0.15)
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
trainSigma  = args.sigma

SNR_range_dB = np.arange( 0.0, 40.1, 2.0 )

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

    m = CommVAE1hot_Aoudia(inVecDim, encDim, h_dim=[64,32,16], n0=trainN0, obj_fn=objFn, sigma=trainSigma)
    h = m.fit(xTrain, epochs=noOfEpochs, batch_size=1000,verbose=0)

    z_mu = m.encode(np.eye(inVecDim))
    sym_pow = np.mean(np.sum(z_mu*z_mu,axis=1))
    print( "Avg. Tx Sym Power:", sym_pow )
    noisePower = sym_pow * 10.0**(-SNR_range_dB/10.0)
    n0_per_comp = noisePower/(2*chUse+2)

    # Adjust pilot power -- this will be used for evaluation
    m.pilot_sym = np.sqrt(sym_pow/encDim) * np.ones(2)
    m.save_model(outPrefix + "_" + timestamp)
    # Clear the graph and load back the model with new pilot power
    # Delete current model
    K.clear_session()
    del m
    # Create new blank model and load back the model with new pilots in graph
    m = CommVAE1hot_Aoudia()
    m.load_model(outPrefix + "_" + timestamp)
    
    err = []
    for n0 in n0_per_comp:
        thisErr = 0
        thisCount = 0
        while thisErr < 500:
            txSym = txSym = np.random.randint(inVecDim, size=1000)
            tx1hot = np.eye(inVecDim)[txSym]
            txTest = m.prenoise_encode(tx1hot)
            rxTest = txTest + np.random.normal(scale=np.sqrt(n0), size=txTest.shape)
            rxDecode = m.postnoise_decode(rxTest)
            rxSym = np.argmax(rxDecode,axis=1)
            thisErr += np.sum(rxSym!=txSym)
            thisCount += 1000
        err.append(thisErr/thisCount)
    blkErr = np.array(err)
    results[timestamp] = (sym_pow, blkErr)

    # print(blkErr)
with open(outPrefix+"_summary.dil", "wb") as f:
    dill.dump(results, f)
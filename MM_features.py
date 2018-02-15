# Feature extraction for MixMe
# 31/7/2017 jjburred for Phonotonic

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.fftpack import dct

# Mel Frequency Cepstral Coefficients
def init_mfcc(stft_pars,fs,mfcc_pars):

    hzPerBin = fs/(2*(stft_pars['numBins_']-1))
    maxBin = math.floor(mfcc_pars['maxFreq']/hzPerBin)

    # create filter bank
    minMel = freq2mel(hzPerBin)
    maxMel = freq2mel(maxBin*hzPerBin)
    melPerBin = (maxMel-minMel)/(mfcc_pars['numFilt']+1)

    centerBinVec = melPerBin*np.ones((1,mfcc_pars['numFilt']+1))
    centerBinVec = np.insert(centerBinVec,0,minMel)

    Vmel2freq = np.vectorize(mel2freq)
    centerBinVec = np.round(Vmel2freq(np.cumsum(centerBinVec))/hzPerBin) - 1

    # init filter bank matrix
    melFB = np.zeros((mfcc_pars['numFilt'],stft_pars['numBins_']))
    for i in range(0,mfcc_pars['numFilt']):
        melFB[i,int(centerBinVec[i]):int(centerBinVec[i+1]+1)] = np.linspace(0,1,centerBinVec[i+1]-centerBinVec[i]+1)
        melFB[i,int(centerBinVec[i+1]):int(centerBinVec[i+2]+1)] = np.linspace(1,0,centerBinVec[i+2]-centerBinVec[i+1]+1)

    if mfcc_pars['plot']:
        for i in range(0,mfcc_pars['numFilt']):
            plt.plot(melFB[i,:])
        plt.show()

    return melFB

def mfcc(spec,melFB,mfcc_pars):

    eps = np.finfo(float).eps
    currMfcc = dct(np.log(np.dot(melFB,spec) + eps),axis=0,type=3,norm='ortho')

    if mfcc_pars['includeEnergy']:
        return currMfcc[:mfcc_pars['numMFCC'],:]
    else:
        return currMfcc[1:mfcc_pars['numMFCC']+1,:]

def freq2mel(freq):
    return 1127.01048*math.log(1+freq/700)

def mel2freq(mel):
    return 700*(math.exp(mel/1127.01048)-1)

def init_chroma(stft_pars,fs):

    # generate one-octave semitone scale
    A3ref = 220  # Hz
    semitone = 2**(1/12)
    C4ref = A3ref*(semitone**3)
    chromaScale = np.log2(C4ref*semitone**np.arange(0,12))-np.floor(np.log2(C4ref*semitone**np.arange(0,12)))

    # compute frequency vector
    freqVec = np.linspace(0,fs/2,stft_pars['numBins_'])
    freqVec = np.delete(freqVec,0)
    freqVec = np.log2(freqVec) - np.floor(np.log2(freqVec))

    # chroma mapping matrix
    chromaMap = np.zeros((len(freqVec),len(chromaScale)))
    for i in range(0,len(chromaScale)):
        chromaMap[:,i] = np.absolute(freqVec-chromaScale[i])

    chromaInd = np.argmin(chromaMap,axis=1)

    return chromaInd

def chroma(spec,chromaInd):

    chromaMat = np.zeros((12,spec.shape[1]))
    spec = np.delete(spec,(0),axis=0)  # remove DC
    # spec = np.log(spec)   # log magnitude?

    for i in range(0,12):
        chromaMat[i,:] = np.mean(spec[chromaInd==i,:],axis=0)

    return chromaMat

def energy(spec):

    return np.sum(spec**2,axis=0)

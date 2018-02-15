# Short Time Fourier Transform
# 31-7-2017 jjburred for Phonotonic

import numpy as np

def stft(waveForm,win,winSize,hopSize,overlap):

    numSamples = waveForm.shape[0]
    numFrames = int(np.floor(numSamples/hopSize)-np.rint(1/(1-overlap)))
    numChannels = waveForm.ndim

    waveForm.shape = (numSamples,numChannels)
    win.shape = (win.shape[0],1)

    win = np.tile(win,(1,numChannels))

    atoms = np.zeros((winSize,numFrames,numChannels))

    pos = 0
    for i in range(0,numFrames):
        atoms[:,i,:] = waveForm[pos:pos+winSize,:]*win
        pos += hopSize

    stft = np.fft.fft(atoms,axis=0)

    return stft

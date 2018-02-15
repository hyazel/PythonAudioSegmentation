# beat detection for MixMe
# 3/8/2017 jjburred for Phonotonic

import os
import numpy as np
import matplotlib.pyplot as plt
import peakutils

def beatDetect(currSpec,stft_pars,beat_pars,currFolder):

    # compute spectral flux
    flux = np.sum(np.diff(currSpec)**2,axis=0)
    flux = flux/np.max(flux)

    # extract periodicity function
    beatFFTsize = int(2**np.ceil(np.log2(len(flux))))
    spec = np.fft.fft(flux, n=beatFFTsize)

    periodicity = np.absolute(spec)

    # detect main periodicity in bpm range
    numFrames = len(flux)
    fs = 1/stft_pars['realHopLength_']
    bpmVecFull = 60*fs/beatFFTsize*np.arange(0,beatFFTsize)
    periodicity = periodicity[bpmVecFull > beat_pars['minBPM']]
    bpmVec = bpmVecFull[bpmVecFull > beat_pars['minBPM']]
    periodicity = periodicity[bpmVec < beat_pars['maxBPM']]
    bpmVec = bpmVec[bpmVec < beat_pars['maxBPM']]

    perDetect = periodicity/np.amax(periodicity)
    perDetect[perDetect<0.7] = 0

    maxInd = peakutils.indexes(perDetect, thres=0.02, min_dist=4)
    maxInd = maxInd[0]

    # maxInd = np.argmax(periodicity)

    currBPM = bpmVec[maxInd]

    if ((currBPM-np.floor(currBPM)<0.2) or (currBPM-np.floor(currBPM)>0.8)):
        currBPM = round(currBPM)

    print('found BPM: {}'.format(currBPM))

    # compute BPM confidence value
    perConf = periodicity/np.amax(periodicity)
    perConf[perConf<0.1] = 0

    peakInd = peakutils.indexes(perConf, thres=0.02, min_dist=4)

    if len(peakInd)==1:
        beatConf = 100
    else:
        peakVal = perConf[peakInd]
        indSort = np.argsort(peakVal)
        indSort = indSort[::-1]
        beatConf = int(round((peakVal[indSort[0]] - peakVal[indSort[1]])*100))

    # plot periodicity detection
    plt.plot(bpmVec,periodicity)
    markerLines = plt.stem([currBPM], [periodicity[maxInd]], '-')
    plt.setp(markerLines, color = 'r', markersize = 8)
    plt.autoscale(enable=True, axis='x', tight=True)
    fig = plt.gcf()
    fig.savefig(os.path.join(currFolder,'periodicity.png'))
    # plt.show()
    plt.clf()

    # place beat markers
    sinFunc = np.cos(2*np.pi*(currBPM/60*np.arange(0,numFrames)/fs))

    sinFunc[sinFunc<0] = 0

    # fine tuning
    fineLim = int(round(60/currBPM*fs))
    corr = np.zeros(fineLim)
    sinOri = sinFunc
    for i in range(0,fineLim):
        corr[i] = np.sum(sinFunc*flux[:len(sinFunc)])
        sinFunc = np.delete(sinFunc,0)
    maxInd = np.argmax(corr)
    sinFunc = np.delete(sinOri,np.arange(0,maxInd))
    sinFunc[sinFunc<0] = 0

    beatInd = peakutils.indexes(sinFunc, thres=0.02, min_dist=4)

    print('found beats: {}'.format(len(beatInd)))

    # # beat time refinement
    # upSample = 10
    # sinFuncHR = np.cos(2*np.pi*(currBPM/60*np.arange(0,numFrames*upSample)/(fs*upSample))+phase*upSample)
    # sinFuncHR = np.delete(sinFuncHR,np.arange(0,2*fineLim*upSample))
    # sinFuncHR = np.insert(sinFuncHR,0,np.zeros((maxInd-2)*upSample))
    # sinFuncHR[sinFuncHR<0] = 0
    # sinFuncHR = sinFuncHR[:oriFluxLength*upSample]
    # beatIndHR = peakutils.indexes(sinFuncHR, thres=0.02, min_dist=4)
    #
    # if beatIndHR[0]!=0:
    #     beatIndHR = np.insert(beatIndHR,0,0)

    # plot flux
    plt.plot(np.arange(0,min(500,len(flux)))/fs,flux[0:min(500,len(flux))])
    plt.plot(np.arange(0,min(500,len(flux)))/fs,sinFunc[0:min(500,len(flux))],color='r')
    # plt.plot(np.arange(0,500*upSample)/(fs*upSample),sinFuncHR[0:500*upSample],color='k')
    # plt.plot(np.arange(0,numFrames)/fs,flux)
    # plt.plot(np.arange(0,numFrames)/fs,sinFunc,color='r')
    # plt.plot(np.arange(0,len(sinFuncHR))/(fs*upSample),sinFuncHR,color='k')

    # markerLines = plt.stem(tempVec[beatInd],np.ones(len(beatInd)), '-')
    # plt.setp(markerLines, color = 'w', markersize = 8)

    plt.autoscale(enable=True, axis='x', tight=True)
    fig.savefig(os.path.join(currFolder,'flux.png'))
    # plt.show()
    plt.clf()

    return beatInd, beatConf
    # return beatInd,beatIndHR,upSample

def exportBeats(beatInd,beatConf,tempVec,currFolder):

    fileName = 'beats_bc'+str(beatConf)+'.txt'
    f = open(os.path.join(currFolder,fileName),'w')
    for b in tempVec:
        f.write("{}\t{}\tb\n".format(b,b))
    f.close()

# MixMe: main script for structure analysis
# 31-7-2017 jjburred for Phonotonic

import os
import numpy as np
import scipy.io.wavfile as wav
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import gridspec
from MM_stft import stft
from MM_features import *
from MM_structure import *
from MM_beat import *
import time
from shutil import copyfile
from sklearn.preprocessing import StandardScaler


# inputFolder = "/Users/jjb/Documents/research/phonotonic/db/test"
inputFolder = "/Users/Phtc-LD/Desktop/Dev/Music/phonoFitTracks/GTTrack" #Apprentissage GTTrack
#inputFolder =  "/Users/Phtc-LD/Desktop/Dev/Music/phonoFitTracks/trackAtester/" #A tester


inputSegmentFolder = "/Users/Phtc-LD/Desktop/Applications/MusicAndSport/mix.me_research/python/GT/Segmentation/"


outputFolder = "/Users/Phtc-LD/Desktop/Applications/MusicAndSport/mix.me_research/python/out"

main_pars = {
    'resolution':   0.04,   # analysis resolution in s
    'mfccWeight':   0.8,    # MFCC weights for clustering (for segmentation, MFCC is always used)
    'chromaWeight': 0.2,    # chroma weights for clustering
    'context':      16,     # segmentation context in beats - combien il regarde autour
    'beatsPerSeg':  16,     # minimum beats per segment
    'partsDetail':  0.6,    # description level for parts clustering
    'debitsDetail': 0.7     # description level for debits clustering
}

# =======================================================

# STFT parameters
stft_pars = {
    'winLength':     main_pars['resolution'], # in s. was 0.04
    'overlapFactor': 0.75
}

# Beat detection parameters
beat_pars = {
    'minBPM': 70,
    'maxBPM': 155
}

# MFCC parameters
mfcc_pars = {
    'numMFCC': 13,
    'numFilt': 40,
    'includeEnergy': 0,
    'maxFreq': 16000,
    'plot': 0
}

# self-similarity matrix parameters
struct_pars = {
    'dist':    'eucl',         # distance: 'eucl', 'exp_cosine', 'cosine', 'corr'
    'cbSize':  main_pars['context'],   # checkerboard size
    'plot': 0,
    'partsDetail':  main_pars['partsDetail'],
    'debitsDetail': main_pars['debitsDetail'],
    'beatsPerSeg':  main_pars['beatsPerSeg'],
    'clustFeat': 'mean_std'    # mean_std, stack, seq_dist
}

# create output folder
expName = time.strftime("MMout_%Y-%m-%d_%H-%M-%S")
expFolder = os.path.join(outputFolder,expName)
if not os.path.exists(expFolder):
    os.makedirs(expFolder)

# copy configuration file
copyfile('./MM_analyze.py',os.path.join(expFolder,'MM_analyze.py'))

for f in os.listdir(inputFolder):
    print("F {}".format(os.path.splitext(f)[0]))
    name = os.path.splitext(f)[0]
    
    str_currFileName = os.path.join(inputFolder,f)
    if str_currFileName.lower().endswith('.wav'):

        #Get semgentation file for ground truth
         
            
        # create current file subfolder
        currFolder = os.path.join(expFolder,f)
        os.makedirs(currFolder)

        # load wave file
        print("Loading wave file: "+str_currFileName)
        (fs,waveformOri) = wav.read(str_currFileName)
        oriLength = waveformOri.shape[0]

        # mix to mono
        if len(waveformOri.shape)>1:
            waveform = np.sum(waveformOri,axis=1)
        else:
            waveform = waveformOri

        # normalize to floats
        waveform = waveform/np.max(np.abs(waveform))

        # compute STFT
        stft_pars['winSize_'] = int(2**np.ceil(np.log2(stft_pars['winLength']*fs)))  # nextpow2
        stft_pars['hopSize_'] = int(np.round(stft_pars['winSize_']*(1-stft_pars['overlapFactor'])))
        stft_pars['realHopLength_'] = stft_pars['hopSize_']/fs

        win = np.hamming(stft_pars['winSize_'])
        currSpec = stft(waveform,win,stft_pars['winSize_'],stft_pars['hopSize_'],stft_pars['overlapFactor'])

        currSpec = np.squeeze(currSpec) # remove singleton dimension for mono STFTs
        currSpec = np.absolute(currSpec[:int(stft_pars['winSize_']/2+1),:])

        print('spectrogram size: {} x {} (bins x frames)'.format(currSpec.shape[0],currSpec.shape[1]))

        beatInd, beatConf = beatDetect(currSpec,stft_pars,beat_pars,currFolder)

        # compute temporal vector
        stft_pars['numFrames_'] = currSpec.shape[1]
        tempVec = np.arange(0,stft_pars['numFrames_'])*stft_pars['realHopLength_']
        tempVec = tempVec[beatInd]
        #
        # # # high-resolution version
        # # tempVecHR = np.arange(0,stft_pars['numFrames_']*upSample)*stft_pars['realHopLength_']/upSample
        # # tempVecHR = tempVecHR[beatIndHR]
        #
        exportBeats(beatInd,beatConf,tempVec,currFolder)

        # compute MFCC
        stft_pars['numBins_'] = currSpec.shape[0]
        melFB = init_mfcc(stft_pars,fs,mfcc_pars)
        currMfcc = mfcc(currSpec,melFB,mfcc_pars)

        # compute chroma
        chromaInd = init_chroma(stft_pars,fs)
        currChroma = chroma(currSpec,chromaInd)

        # compute energy
        currEnergy = energy(currSpec)
        
        if main_pars['mfccWeight']==0:
            clustFeatMat = currChroma
        elif main_pars['chromaWeight']==0:
            clustFeatMat = currMfcc
        else:
            currChromaW = currChroma * main_pars['chromaWeight']
            currMfccW = currMfcc * main_pars['mfccWeight']
            clustFeatMat = np.concatenate((currMfccW,currChromaW),axis=0)

        segFeatMat = currMfcc         # feature matrix for segmentation

        # normalize feature matrices
        scaler = StandardScaler()
        segFeatMat = scaler.fit_transform(np.transpose(segFeatMat))
        segFeatMat = np.transpose(segFeatMat)

        scaler = StandardScaler()
        clustFeatMat = scaler.fit_transform(np.transpose(clustFeatMat))
        clustFeatMat = np.transpose(clustFeatMat)

        # median filter
        # segFeatMat = sp.signal.medfilt(segFeatMat,5)
        # clustFeatMat = sp.signal.medfilt(clustFeatMat,5)

        # quantize seg matrix to beats
        numBeats = len(beatInd)
        segFeatMatQ = np.zeros((segFeatMat.shape[0],numBeats-1))
        for i in range(0,numBeats-1):
            segFeatMatQ[:,i] = np.mean(segFeatMat[:,beatInd[i]:beatInd[i+1]],axis=1)

        # quantize clust matrix to beats
        clustFeatMatQ = np.zeros((clustFeatMat.shape[0],numBeats-1))
        for i in range(0,numBeats-1):
            clustFeatMatQ[:,i] = np.mean(clustFeatMat[:,beatInd[i]:beatInd[i+1]],axis=1)
        
        # quantize energy
        energyQ = np.zeros((numBeats-1,1))
        for i in range(0,numBeats-1):
            energyQ[i] = np.mean(currEnergy[beatInd[i]:beatInd[i+1]])

        print('seg. feat. matr. size: {} x {} (dim x beats)'.format(segFeatMatQ.shape[0],segFeatMatQ.shape[1]))
        
        # self-similarity matrix
        currSSM = SSM(segFeatMatQ,struct_pars,currFolder)

        # segmentation based on SSM
        boundaries = SSM_segment(currSSM,struct_pars,currFolder,energyQ)
        
        temp = tempVec[boundaries]

        # export segments
        f = open(os.path.join(currFolder,'segments.txt'),'w')
        for i,b in enumerate(temp):
            f.write("{}\t{}\t{}\n".format(b,b,i))
        f.close()

        # cluster segments
        #secIDs,subsecIDs,boundaries,clustVar,clusterKmeansPartie,clusterKmeansDebit,clustFeat = segCluster(clustFeatMatQ,boundaries,struct_pars,currFolder)
        
        # sort debits by intensity
        #subsecIDs,globalIntensity = sortDebits(currSpec,beatInd,boundaries,secIDs,subsecIDs)
        #newSubSecIDs,globalIntensity = sortDebits(currSpec,beatInd,boundaries,clusterKmeansPartie,clusterKmeansDebit)

        # export data
        #export(boundaries,secIDs,subsecIDs,tempVec,currFolder,waveformOri,fs,globalIntensity,clustVar)
        #export(boundaries,clusterKmeansPartie,newSubSecIDs,tempVec,currFolder,waveformOri,fs,globalIntensity,clustVar)
        
        
        
        #TEST LAURENT
            
        
        str = "/Users/Phtc-LD/Desktop/Dev/Music/phonoFitTracks/" + name + ".png"
        str2 = "/Users/Phtc-LD/Desktop/Dev/Music/phonoFitTracks/" + name + "-R.png"
        #[clusterKmeansPartie,cCluster,inertia] = MyKmeans(5,clustFeat[1:19],None)
        
        
        fig = plt.figure()
        #plt.ylim([0,1])
        plt.stem(secIDs)
        #plt.savefig(str)
        plt.show()
        plt.close()
        
        
        fig = plt.figure()
        #plt.ylim([0,1])
        plt.stem(clusterKmeansPartie)
        #plt.savefig(str)
        plt.show()
        plt.close()
        
        
        #"""
        segmentPath = inputSegmentFolder + name + '.txt'
        maSegmentsGT = []
        with open(segmentPath) as fl: 
            for line in fl: 
                l = line.split("\t") 
                maSegmentsGT.append(float(l[0]))
                
        
        
        #"""
        
        """
        segmentPathATester = inputFolder + name + '.txt'
        maSegmentsGT = []
        with open(segmentPathATester) as fl: 
            for line in fl: 
                l = line.split("\t") 
                maSegmentsGT.append(float(l[0]))
        """
        
        compteurGTSegment = 0
        for k in range(0,len(maSegmentsGT)):
            for m in range(0,len(temp)):
                if(temp[m]-0.1<maSegmentsGT[k]<temp[m]+0.1):
                    compteurGTSegment = compteurGTSegment + 1
                    break
        
        file = open("/Users/Phtc-LD/Desktop/Dev/Music/phonoFitTracks/segmentsGT.txt","a") 
        file.write("{}\t{}\n".format(name,100*compteurGTSegment/len(temp)))
        file.close()
        
        
       
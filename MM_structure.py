# Structure segmentation for MixMe
# 31/7/2017 jjburred for Phonotonic

import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import scipy as sp
import peakutils
import scipy.io.wavfile as wav
from Kmeans import *
import math

def dist_nico(X, Y):

   edist = 0.0
   output_dist = 0.0
   count = 0
   for i in range (1,len(X)):
       edist = math.pow(Y[i] - X[i], 2)
       if (edist <=40):
           count += 1
           output_dist += edist
   return output_dist

def SSM(featMat,struct_pars,currFolder):

    numFrames = featMat.shape[1]

    SSM = np.zeros((numFrames,numFrames))

    # compute symmetric frame-by-frame similarities
    for i in range(0,numFrames):
        # print(i)
        for j in range(i,numFrames):

            # euclidean distance
            if struct_pars['dist']=='eucl':
                SSM[i,j] = -norm(featMat[:,i]-featMat[:,j])

            # cosine distance
            elif struct_pars['dist']=='cosine':
                SSM[i,j] = (np.dot(featMat[:,i],featMat[:,j])/(norm(featMat[:,i])*norm(featMat[:,j])))**2

            # exponential cosine distance
            elif struct_pars['dist']=='exp_cosine':
                    SSM[i,j] = np.exp(np.dot(featMat[:,i],featMat[:,j])/(norm(featMat[:,i])*norm(featMat[:,j]))-1)

            # correlation distance (like scipy.spatial.distance.correlation)
            elif struct_pars['dist']=='corr':
                mu1 = featMat[:,i].mean()
                mu2 = featMat[:,j].mean()
                s1 = featMat[:,i] - mu1
                s2 = featMat[:,j] - mu2
                SSM[i,j] = ((np.dot(s1,s2))/(norm(s1)*norm(s2)))**2

    # mirror (no need, just for plotting)
    for i in range(1,numFrames):
        for j in range(i-1,-1,-1):
            SSM[i,j] = SSM[j,i]

    # min-max normalization
    tMin = np.amin(SSM)
    tMax = np.amax(SSM)
    SSM = (SSM-tMin)/(tMax-tMin)

    # save SSM image
    plt.figure(1)
    plt.imshow(SSM,interpolation="nearest")

    fig = plt.gcf()
    if struct_pars['plot']:
        plt.show()
    fig.savefig(os.path.join(currFolder,'SSM.png'))
    plt.clf()

    return SSM

def SSM_segment(SSM,struct_pars,currFolder,energyQ):

    # direct clustering?

    # direct k-means on SSM
    # cl = KMeans(n_clusters=3)

    # direct DBSCAN on SSM
    # cl = DBSCAN(eps=0.5, min_samples=10)
    # cl.fit(SSM)

    # frameLabels = cl.labels_
    # print(frameLabels)

    # return frameLabels

    # novelty curve with checkerboard
    cbSize = struct_pars['cbSize']
    CB = np.kron(np.array([[1,-1],[-1,1]]),np.ones((cbSize,cbSize)))
    M = CB.shape[0]

    # pad SSM edges by repeating data
    numFrames = SSM.shape[0]
    SSM = np.c_[np.tile(SSM[:,[0]],M),SSM,np.tile(SSM[:,[-1]],M)]
    SSM = np.r_[np.tile(SSM[0,:],(M,1)),SSM,np.tile(SSM[-1,:],(M,1))]

    numFramesPad = SSM.shape[0]
    novelty = np.zeros(numFramesPad)

    for i in range(M//2, numFramesPad-M//2+1):
        novelty[i] = np.sum(SSM[i - M//2:i + M//2, i-M//2:i + M//2] * CB)

    # Normalize
    novelty += novelty.min()
    novelty /= novelty.max()
    
    # median filt?
    novelty = sp.signal.medfilt(novelty,5)

    boundaries = peakutils.indexes(novelty, thres=0.02, min_dist=struct_pars['beatsPerSeg'])

    # cut novelty
    novelty = np.delete(novelty,np.arange(0,M))
    novelty = np.delete(novelty,np.arange(len(novelty)-M,len(novelty)))

    boundaries -= M
    boundaries = boundaries[boundaries<=len(novelty)]
    if boundaries[-1]==len(novelty):
        boundaries[-1] -= 1
    
    # find downbeat
    #"""
    boundAmpl = novelty[boundaries]
    boundAmpl[:2] = 0
    boundAmpl[-2:] = 0
    indSort = np.argsort(boundAmpl)  # highest novelty values
    indSort = indSort[::-1]
    
    # boundOri = boundaries

    numPeaks = 15
    beatsPerSeg = struct_pars['beatsPerSeg']   # was 8
    downbeatList = []
    common = []
    for ind in indSort[:numPeaks]:
        afterVec  = boundaries[ind]+beatsPerSeg*np.arange(0,int((numFrames-boundaries[ind])/beatsPerSeg+1))
        beforeVec = boundaries[ind]-beatsPerSeg*np.arange(1,int(boundaries[ind]/beatsPerSeg)+1)
        downbeatVec = np.concatenate((afterVec,beforeVec))
        #downbeatVec = afterVec
        downbeatList.append(downbeatVec)
        common.append(len(np.intersect1d(downbeatVec,boundaries)))

    bestInd = np.argmax(np.asarray(common))
    boundaries = np.unique(np.asarray(downbeatList[bestInd]))
    #"""
    if boundaries[0]!=0:
        boundaries = np.insert(boundaries,0,0)
    
    boundaries = boundaries[boundaries>=0]
    boundaries = boundaries[boundaries<numFrames]

    # discard edge segments with very low energy
    discardInd = []
    enThresh = 500
    if np.mean(energyQ[boundaries[0]:boundaries[1]])<enThresh:
        boundaries = np.delete(boundaries,0)
    if np.mean(energyQ[boundaries[-2]:boundaries[-1]])<enThresh:
        boundaries = np.delete(boundaries,-1)

    # discard short edge segments
    if (boundaries[1]-boundaries[0])<2:
        boundaries = np.delete(boundaries,0)
    if (boundaries[-1]-boundaries[-2])<2:
        boundaries = np.delete(boundaries,-1)

    plt.autoscale(enable=True, axis='x', tight=True)
    plt.plot(novelty)
    # markerLines = plt.stem(boundOri, novelty[boundOri], '-')
    markerLines = plt.stem(boundaries, novelty[boundaries], '-')
    plt.setp(markerLines, color = 'r', markersize = 8)
    fig = plt.gcf()
    fig.savefig(os.path.join(currFolder,'novelty.png'))
    # plt.show()
    plt.clf()

    print('found segments: {}'.format(len(boundaries)))

    return boundaries

def segCluster(featMat,boundaries,struct_pars,currFolder):

    # featMat: dim X obs

    if (struct_pars['clustFeat']=='mean_std'):

        segCentroids = np.zeros((len(boundaries),featMat.shape[0]))
        segStd = np.zeros((len(boundaries),featMat.shape[0]))
        boundaries = np.append(boundaries,featMat.shape[1])

        for i in range(1,len(boundaries)):
            segCentroids[i-1,:] = np.mean(featMat[:,boundaries[i-1]:boundaries[i]],axis=1)
            segStd[i-1,:] = np.std(featMat[:,boundaries[i-1]:boundaries[i]],axis=1)

        clustFeat = np.concatenate((segCentroids,segStd),axis=1)

        scaler = StandardScaler()
        clustFeat = scaler.fit_transform(clustFeat)

        boundaries = np.delete(boundaries,-1)
        
        

    elif (struct_pars['clustFeat']=='stack'):

        clustFeat = np.zeros((len(boundaries),featMat.shape[0]*struct_pars['beatsPerSeg']))
        numDim = featMat.shape[0]

        for i in range(1,len(boundaries)):
            currBeats = boundaries[i] - boundaries[i-1]
            clustFeat[i,:numDim*currBeats] = np.reshape(featMat[:,boundaries[i-1]:boundaries[i]],(numDim*currBeats,1),order='F').transpose()

    elif (struct_pars['clustFeat']=='seq_dist'):

        clustFeat = np.zeros((len(boundaries),len(boundaries)))    # acutally, distance matrix

        for i in range(1,len(boundaries)):
            for j in range(1,len(boundaries)):
                currBeats = min((boundaries[i]-boundaries[i-1],boundaries[j]-boundaries[j-1]))
                clustFeat[i,j] = np.sum((featMat[:,boundaries[i-1]:(boundaries[i-1]+currBeats)]-featMat[:,boundaries[j-1]:(boundaries[j-1]+currBeats)])**2)

        clustFeat = squareform(clustFeat)

    #Kmeans
    [clusterKmeansPartie,cCluster,inertia] = MyKmeans(5,clustFeat,None)
    [clusterKmeansDebit,cCluster,inertia] = MyKmeans(10,clustFeat,None)
    
    # agglomerative clustering
    Z = linkage(clustFeat, method='complete', metric='euclidean')
    #Z = linkage(clustFeat, method='complete', metric=dist_nico)
    
    maxDistance = Z[-1,2]

    secCut = maxDistance*(1-struct_pars['partsDetail'])       # dendogram cut for sections
    subsecCut = maxDistance*(1-struct_pars['debitsDetail'])   # dendogram cut for subsections (débits)

    #secIDs = fcluster(Z, secCut, criterion='distance')
    secIDs = fcluster(Z, secCut, criterion='distance')
    subsecIDs = fcluster(Z, subsecCut, criterion='distance')

    dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
    fig = plt.gcf()
    fig.savefig(os.path.join(currFolder,'dendogram.png'))
    # plt.show()
    plt.clf()
    
    # compute main cluster variances
    numClusters = len(np.unique(secIDs))
    if (struct_pars['clustFeat']=='seq_dist'):
        clustVar = np.zeros((numClusters,1))
    else:
        clustVar = np.zeros((numClusters,1))
        for i in range(0,numClusters):
            clustVar[i] = np.mean(np.var(clustFeat[secIDs==(i+1),:],axis=0))
    
    #Var sur les kmeans
    numClusters = len(np.unique(clusterKmeansPartie))
    clustVar = np.zeros((numClusters,1))
    
    print(numClusters)
    print("****")
    for i in range(0,len(clusterKmeansPartie)):
        print(clusterKmeansPartie[i])
    print("*******")
    for i in range(0,numClusters):
        print("i : {} -: {}".format(i,clustFeat[clusterKmeansPartie==(i),:]))
        print("******")
        
        clustVar[i] = np.mean(np.var(clustFeat[clusterKmeansPartie==(i),:],axis=0))
    
            
    # # merge boundaries with same consecutive labels
    # lastSec = -1
    # lastSub = -1
    # keepBoundaries = []
    # for i in range(0,len(boundaries)):
    #     if (secIDs[i]!=lastSec or subsecIDs[i]!=lastSub):
    #         lastSec = secIDs[i]
    #         lastSub = subsecIDs[i]
    #         keepBoundaries.append(i)
    #
    # boundaries = boundaries[keepBoundaries]
    # secIDs = secIDs[keepBoundaries]
    # subsecIDs = subsecIDs[keepBoundaries]

    return secIDs,subsecIDs,boundaries,clustVar,clusterKmeansPartie,clusterKmeansDebit,clustFeat

def sortDebits(currSpec,beatInd,boundaries,secIDs,subsecIDs):

    intensity = []
    for i in range(0,len(boundaries)-1):
        intensity.append(np.mean(currSpec[:,beatInd[boundaries[i]]:beatInd[boundaries[i+1]]]))
    intensity.append(0)

    # normalize intensity
    intensity = np.asarray(intensity)/np.amax(np.asarray(intensity))

    # sort inside each section
    uniqueSecIDs = np.unique(secIDs)
    newSubSecIDs = np.zeros(len(subsecIDs),dtype=int)

    for i in uniqueSecIDs:
        currSubSecIDs = np.unique(subsecIDs[secIDs==i])
        meanInt = []
        for j in currSubSecIDs:
            meanInt.append(np.mean(intensity[subsecIDs==j]))
        currInd = np.argsort(meanInt)
        for j in range(0,len(currSubSecIDs)):
            newSubSecIDs[subsecIDs==currSubSecIDs[currInd[j]]] = j

    return newSubSecIDs,intensity


def export(boundaries,secIDs,subsecIDs,tempVec,currFolder,waveform,fs,globalIntensity,clustVar):

    boundaries = tempVec[boundaries]

    # # export segments
    # f = open(os.path.join(currFolder,'segments.txt'),'w')
    # for i,b in enumerate(boundaries):
    #     f.write("{}\t{}\t{}\n".format(b,b,i))
    # f.close()
    
    # generate section labels from IDs
    letters = 'ABCDEFGHIJKLMNOPQRESTUVWXYZ'
    letDict = np.zeros(len(letters),dtype=str)
    _,ind = np.unique(secIDs,return_index=True)
    uniqueIDs = secIDs[np.sort(ind)]
    for i,u in enumerate(uniqueIDs):
        letDict[u] = letters[i]
    secLabels = letDict[secIDs]

    # export labels
    f = open(os.path.join(currFolder,'labels.txt'),'w')
    for b,sl,ssl,inten in zip(boundaries,secLabels,subsecIDs,globalIntensity):
        f.write("{}\t{}\t{}{}_{}\n".format(b,b,sl,ssl,int(inten*100)))
    f.close()

    if len(waveform.shape)==1:
        waveform = np.reshape(waveform,(-1,1))

    # export sound segments
    segFolder = os.path.join(currFolder,'debits')
    os.makedirs(segFolder)
    for i in range(0,len(boundaries)-1):
        firstSample = int(np.floor(boundaries[i]*fs))
        lastSample = int(np.floor(boundaries[i+1]*fs))-1
        currSeg = waveform[firstSample:lastSample,:]
        currName = str(i)+'_'+str(secLabels[i])+str(subsecIDs[i])+'_v'+str(int(clustVar[secIDs[i]-1]*100))+'_i'+str(int(globalIntensity[i]*100))+'.wav'
        # de-normalize??
        wav.write(os.path.join(segFolder,currName),fs,currSeg)

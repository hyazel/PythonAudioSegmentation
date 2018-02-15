#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:40:58 2017

@author: Phtc-LD
"""
import numpy as np
from collections import Counter
from acp import *
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from PFMath import *

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

def MyKmeans(nb,matrice,initMatrix):
    if(initMatrix!=None):
        print("initialisation manuelle")
        est=KMeans(n_clusters=nb,init=initMatrix)
    else:
        print("initialisation automatique")
        est=KMeans(n_clusters=nb)
        
        
    est.fit(matrice)
    classe=est.labels_
    inertia=est.inertia_
    return classe,est.cluster_centers_,inertia
    
def CalculDistanceIntra(aa,cs,aaCs):
    print("Début : calcul distance intra")
    maDistanceI = np.zeros([len(aaCs),1])
    maPoids = Counter(cs)
    print("Répartition des clusters")
    print(maPoids)
    for n in range(0,len(aa)):
        maDistanceI[cs[n]] = maDistanceI[cs[n]] + DEucl(aa[n],aaCs[cs[n]])
    
    for n in range(0,len(maDistanceI)):
        maDistanceI[n] = maDistanceI[n]/maPoids[n]

    return maDistanceI
    
    

    
def CalculMatriceSimilarite(maCs,aa,cs):
    maSimilarity = np.zeros([len(maCs),len(maCs)])
    maDistIntra = CalculDistanceIntra(aa,cs,maCs)
    
    print("calcul matrice distance intra")
    print(maDistIntra)
    
    for n in range(0,len(maCs)):
        for o in range(0,len(maCs)):
            #maSimilarity[n][o] = maDistIntra[n]/DEucl(maCs[n],maCs[o])
            #maSimilarity[n][o] = DEucl(maCs[n],maCs[o])/maDistIntra[n]
            maSimilarity[n][o] = DEucl(maCs[n],maCs[o])
    
    print("medianne distance inter cluster")
    print(print(np.median(maSimilarity,axis=1)))
    
    print("moyenne distance inter cluster")
    print(print(np.mean(maSimilarity,axis=1)))
    
    [tab1,tab2,index90] = myPCA(pd.DataFrame(maCs),len(maCs[0]))
    
    return maSimilarity,maDistIntra
    
def compareWithLabel(cs,lLabel):
    nbCls = len(Counter(cs))
        
    mLabelCluster = [[]]

    for i in range(0,nbCls):
        mLabelCluster.append([])
        
    for i in range(0,len(cs)):
        mLabelCluster[cs[i]].append(lLabel[i])
        
    for i in range(0,nbCls):
        print(Counter(mLabelCluster[i]))
        
    return mLabelCluster
        

    
    

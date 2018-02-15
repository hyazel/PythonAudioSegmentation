#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:02:25 2017

@author: Phtc-LD
"""
import numpy as np

def NormalisationMfcc(m):
    min = np.min(m)
    max = np.max(m)
    
    for i in range(0,len(m)):
        for j in range(0,len(m[0])):
            m[i][j] = (m[i][j] -  min)/(max - min)
    
    return m
    
def minWithFiltre(maIntra,counter,seuil,cluster):
    minIndex = -1
    minIndex2 = -1
    
    listIndex = []
    for i in range(0,len(maIntra)):
        if(counter[i]>8):
            listIndex.append([i,maIntra[i][0]])
    
    listIndex = sorted(listIndex, key=lambda tup: tup[1])

    minIndex = listIndex[0][0]      
            
    sum = 0       
    for i in range(0,7):
        if(cluster[i]==minIndex):
            sum = sum + 1
        
    if(sum>2):
        minIndex = listIndex[1][0]


    print("listIndex {}".format(listIndex))
    return minIndex
    
def ModifWrongValues(a):
    for i in range(0,len(a)):
        if(np.sum(a[i]) > 10):
            for j in range(0,len(a[i])):
                a[i][j] = 0
            
    return a
    
def seuilPitch(a,seuil):
    for i in range(0,len(a)):
        for j in range(0,len(a[i])):
            if(a[i][j]<seuil):
                a[i][j] = 0
            else:
                a[i][j] = 1
    return a
    
def TailleBoucle(cluster):
    taille = 0
    for i in range(0,len(cluster) - 2):
        if(cluster[i] != cluster[i+1] and cluster[i+1] != cluster[i+2]):
            taille = taille + 1
      
     
    if taille/len(cluster) > 0.25:
        taille=16
    else:
        taille=8
    print("taille boucle : {}".format(taille)) 
    return taille
def DEucl(a1,a2):
    return np.linalg.norm(a1-a2)
    
def kl(p, q):
   meanP = np.mean(p)
   meanQ = np.mean(q)
   varP = np.var(p)
   varQ = np.var(q)
   
   return varP/varQ + varQ/varP + (meanP - meanQ)*(1/varP + 1/varQ)
   
    
    
    
def MatrixSimi(ma):
    maSimilarity = np.zeros([len(ma),len(ma)])
    
    for i in range(0,len(ma)):
        for j in range(0,len(ma)):
            
            maSimilarity[i][j] = kl(ma[i],ma[j])
            
    return maSimilarity     
            
            
    
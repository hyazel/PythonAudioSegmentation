#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:16:49 2017

@author: Phtc-LD
"""

from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def myPCA(df,nbC,clusters=None):
    
    # Normalize data
    #df_norm = (df - df.mean()) / df.std()
    
    
    # PCA
    pca = PCA(n_components=nbC,tol=0.95)
    pca_res = pca.fit_transform(df)
    # Ebouli
    ebouli = pd.Series(pca.explained_variance_ratio_)
    ebouli.plot(kind='bar', title="Ebouli des valeurs propres")
    
    sumCum = 0
    index = 0
    index90 = len(ebouli)
    for num in ebouli:
        sumCum = sumCum +  num
        index+=1
        if(sumCum>0.9 and index90>index):
            index90= index
            
        #print ("{} : {}".format(index,sumCum))
        
    print("index 90% : {}".format(index90))
        
    
    
    plt.show()
    # Circle of correlations
    # http://stackoverflow.com/a/22996786/1565438
    coef = np.transpose(pca.components_)
    cols = ['PC-'+str(x) for x in range(len(ebouli))]
    pc_infos = pd.DataFrame(coef, columns=cols, index=df.columns)
    circleOfCorrelations(pc_infos, ebouli)
    plt.show()
    # Plot PCA
    index = 0

    dat = pd.DataFrame(pca_res, columns=cols)
        
    if isinstance(clusters, np.ndarray):
        fig = plt.figure()
        #ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
        for clust in set(clusters):
            colors = ['gold','purple','black','cyan','yellow','bisque','snow','gainsboro','navy','salmon']
            #labels= ['gold','pink','darkgreen','red','y','y','y','y','y','y','y']
            plt.scatter(dat["PC-0"][clusters==clust],dat["PC-1"][clusters==clust],color=colors[clust],label=labels[clust])
            plt.scatter(dat["PC-0"][clusters==clust],dat["PC-1"][clusters==clust],color=colors[clust])
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            
    else:
        label = list(range(len(dat)))
        for ii in range (0,len(dat)):
            plt.scatter(dat["PC-0"][ii],dat["PC-1"][ii])
         
        for i, lb in enumerate(label):
            plt.annotate(lb, (dat["PC-0"][i],dat["PC-1"][i]))
    
    plt.xlabel("PC-0 (%s%%)" % str(ebouli[0])[:4].lstrip("0."))
    plt.ylabel("PC-1 (%s%%)" % str(ebouli[1])[:4].lstrip("0."))
    plt.title("PCA")
    
    """label = np.zeros(len(df))
    for i in range(0, len(df)):
        label[i] = i
        
    for i, txt in enumerate(label):
       plt.annotate(txt,(dat["PC-0"][i],dat["PC-1"][i]))
      """  
    plt.show()
                
    return pca.components_,pca_res,index90
    
def circleOfCorrelations(pc_infos, ebouli):
	plt.Circle((0,0),radius=10, color='g', fill=False)
	circle1=plt.Circle((0,0),radius=1, color='g', fill=False)
	fig = plt.gcf()
	fig.gca().add_artist(circle1)
	for idx in range(len(pc_infos["PC-0"])):
		x = pc_infos["PC-0"][idx]
		y = pc_infos["PC-1"][idx]
		plt.plot([0.0,x],[0.0,y],'k-')
		plt.plot(x, y, 'rx')
		plt.annotate(pc_infos.index[idx], xy=(x,y))
	plt.xlabel("PC-0 (%s%%)" % str(ebouli[0])[:4].lstrip("0."))
	plt.ylabel("PC-1 (%s%%)" % str(ebouli[1])[:4].lstrip("0."))
	plt.xlim((-1,1))
	plt.ylim((-1,1))
	plt.title("Circle of Correlations")
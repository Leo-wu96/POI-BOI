import random
import pickle
import numpy as np
import math
import pandas
import tensorflow as tf
import scipy.io as io
from sklearn.cluster import *
from sklearn import preprocessing,metrics


random.seed(2021)
sampleNum = -1




def vecClusterAnalysis(path):
    print('---------------------------------')
    
    trVecs = np.load(path)
    perm = np.arange(len(trVecs))
    np.random.shuffle(perm)
    trVecs = trVecs[perm]
    s = []
    e = []
    for i in range(2,100,2):
        km = KMeans(n_clusters=i,init= 'k-means++', random_state=2021)
        all_clusters = km.fit(trVecs).labels_.tolist()

        clusters = all_clusters[:sampleNum]

        s.append(metrics.silhouette_score(trVecs[:sampleNum],clusters,metric='euclidean'))
        e.append(km.inertia_)
    io.savemat('clusters.mat', {'slihouette':np.asarray(s),'elbow':e})
    
if __name__ == '__main__':
    vecClusterAnalysis('path')
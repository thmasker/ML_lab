import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.decomposition import PCA



FEATURES_FILE = os.path.join('.', 'data', 'processed', 'T2_features.csv')


print(f"Loading extracted features from {FEATURES_FILE}")
df = pd.read_csv(FEATURES_FILE)
df = df.dropna()


##  [ Normalization ]
scaler = preprocessing.MinMaxScaler()
df_norm = scaler.fit_transform(df)


## [ PCA Estimation ]
estimator = PCA(n_components = 2)
X_pca = estimator.fit_transform(df_norm)

print(estimator.explained_variance_ratio_)

## [ DBScan ]
# 1. Compute the similarity matrix
import sklearn.neighbors

dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
matsim = dist.pairwise(df_norm)

# 1.1 Visualization
import seaborn as sns

ax = sns.heatmap(matsim,vmin=0, vmax=1)

from sklearn.neighbors import kneighbors_graph
import numpy

minPts=3

A = kneighbors_graph(df_norm, minPts, include_self=False)
Ar = A.toarray()

seq = []
for i,s in enumerate(df_norm):
    for j in range(len(df_norm)):
        if Ar[i][j] != 0:
            seq.append(matsim[i][j])
            
seq.sort()
# establecer intervalo ejes
fig = plt.figure()
ax = fig.gca()
ax.set_xticks(numpy.arange(0, 150, 20))
ax.set_yticks(numpy.arange(0, 2.5, 0.25))

plt.plot(seq)


plt.show()

from sklearn.cluster import DBSCAN

for eps in numpy.arange(0.25, 0.85, 0.10):
  db = DBSCAN(eps, min_samples=minPts).fit(df_norm)
  core_samples_mask = numpy.zeros_like(db.labels_, dtype=bool)
  core_samples_mask[db.core_sample_indices_] = True
  labels = db.labels_
  n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
  n_outliers = list(labels).count(-1)
  print ("%6.2f, %d, %d" % (eps, n_clusters_, n_outliers))

print(labels)

db = DBSCAN(eps=0.55, min_samples=minPts).fit(df_norm)
labels = db.labels_

#plotting orginal points with color related to label
fig, ax = plt.subplots()

plt.scatter(X_pca[:,0], X_pca[:,1], c=labels,s=50, cmap="Accent")
# anottation
for i in range(0,len(X_pca)):
    if labels[i] == -1: 
      ax.annotate(df.iloc[i,:].name[0:3], (X_pca[i,0], X_pca[i,1]))
plt.grid()
plt.show()

df['dbscan_group'] = labels

print(df[df['dbscan_group'] == -1])
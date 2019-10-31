import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from sklearn.decomposition import PCA


FEATURES_FILE = os.path.join('.', 'T2_features.csv')


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

## [ Hierarchical Clustering ]
# 1. Compute the similarity matrix
import sklearn.neighbors

dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
matsim = dist.pairwise(df_norm)

# 2. Building the Dendrogram	
from scipy import cluster

clusters = cluster.hierarchy.linkage(matsim, method = 'complete')
cluster.hierarchy.dendrogram(clusters, color_threshold=10)
plt.show()

cut = 10 # !!!! ad-hoc
labels = cluster.hierarchy.fcluster(clusters, cut , criterion = 'distance')
print('Number of clusters %d' % (len(set(labels))))

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)


fig, ax = plt.subplots()
plt.xlim(-1, 2)
plt.ylim(-0.5, 1)

for i in range(len(X_pca)):
    plt.text(X_pca[i][0], X_pca[i][1], 'x', color=colors[labels[i]])  
    
ax.grid(True)
fig.tight_layout()
plt.show()

# tratamos el cluster -1 como cluster de outliers
from sklearn import metrics

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(df_norm, labels))

df['group'] = labels
res = df.groupby(('group')).size()
print(res)
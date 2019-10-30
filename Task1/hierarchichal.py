import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("T2_features.csv")
df = df.dropna()

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
datanorm = min_max_scaler.fit_transform(df)

from sklearn.decomposition import PCA

estimator = PCA (n_components = 2)
X_pca = estimator.fit_transform(datanorm)

import numpy

print(estimator.explained_variance_ratio_)

#3. Hierarchical Clustering
# 3.1. Compute the similarity matrix
import sklearn.neighbors

dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
matsim = dist.pairwise(datanorm)

# 3.2. Building the Dendrogram	
from scipy import cluster

clusters = cluster.hierarchy.linkage(matsim, method = 'complete')
cluster.hierarchy.dendrogram(clusters, color_threshold=10)
plt.show()

cut = 10 # !!!! ad-hoc
labels = cluster.hierarchy.fcluster(clusters, cut , criterion = 'distance')
print('Number of clusters %d' % (len(set(labels))))

colors = numpy.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = numpy.hstack([colors] * 20)


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
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(datanorm, labels))

df['group'] = labels
res = df.groupby(('group')).size()
print(res)
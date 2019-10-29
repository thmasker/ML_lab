import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("T2_features.csv")
df = df.dropna()

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
df_scaled = scaler.fit_transform(df)

#2. PCA Estimation
from sklearn.decomposition import PCA

estimator = PCA(n_components = 2)
X_pca = estimator.fit_transform(df_scaled)

# parameters
init = 'random' # initialization method 

# to run 10 times with different random centroids 
# to choose the final model as the one with the lowest SSE
iterations = 10

# maximum number of iterations for each single run
max_iter = 300 

# controls the tolerance with regard to the changes in the 
# within-cluster sum-squared-error to declare convergence

tol = 1e-04 

 # random seed
random_state = 0

from sklearn.cluster import KMeans
from sklearn import metrics

distortions = []
silhouettes = []

for i in range(2, 11):
    km = KMeans(i, init, n_init = iterations ,max_iter= max_iter, tol = tol,random_state = random_state)
    labels = km.fit_predict(X_pca)
    distortions.append(km.inertia_)
    silhouettes.append(metrics.silhouette_score(X_pca, labels))

# plt.plot(range(2,11), distortions, marker='o')
# plt.xlabel('K')
# plt.ylabel('Distortion')
# plt.show()

# plt.plot(range(2,11), silhouettes , marker='o')
# plt.xlabel('K')
# plt.ylabel('Silhouette')
# plt.show()

k = 4

km = KMeans(k, init, n_init = iterations ,
            max_iter= max_iter, tol = tol, random_state = random_state)

y_km = km.fit_predict(X_pca)

from sklearn import metrics

print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X_pca, y_km))
      
print('Distortion: %.2f' % km.inertia_)

# #plotting orginal points with color related to label
# plt.scatter(X_pca[:,0], X_pca[:,1], c=km.labels_,s=50)
# # plotting centroids
# plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], c='blue',s=50)
# plt.grid()
# plt.show()

df['kmeans_group'] = km.labels_

res = df[['AccelerometerStat_x_FIRST_VAL_FFT','AccelerometerStat_y_FIRST_VAL_FFT','AccelerometerStat_z_FIRST_VAL_FFT','kmeans_group']].groupby(('kmeans_group')).mean()
# res.plot(kind='bar', legend=True)
# plt.show()
res.to_excel('KMeans.xlsx')
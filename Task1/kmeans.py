import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics

#----------------------------------------------------------
# K-means parameters
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
#----------------------------------------------------------

FEATURES_FILE = os.path.join('.', 'T2_features.csv')


print(f"Loading extracted features from {FEATURES_FILE}")
df = pd.read_csv(FEATURES_FILE)
df = df.dropna()


##  [ Normalization ]
scaler = preprocessing.StandardScaler()
df_scaled = scaler.fit_transform(df)


## [ PCA Estimation ]
estimator = PCA(n_components = 2)
X_pca = estimator.fit_transform(df_scaled)



## [ Silhouette & distortion checking ]
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

# Interpretation:
#   - k=4 seems like the most balanced choice, since
#       it provides a high silhouette and a low distortion


## [ K-Means ]
k = 4

km = KMeans(k, init, n_init = iterations ,
            max_iter= max_iter, tol = tol, random_state = random_state)

y_km = km.fit_predict(X_pca)

print("Silhouette Coefficient: {:0.3f}".format(metrics.silhouette_score(X_pca, y_km)))
print('Distortion: {:.2f}'.format(km.inertia_))


## [ Plot ]

# #plotting orginal points with color related to label
# plt.scatter(X_pca[:,0], X_pca[:,1], c=km.labels_,s=50)
# # plotting centroids
# plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], c='blue',s=50)
# plt.grid()
# plt.show()

# Interpretation:
#   - The phone is usually resting in a table (probably blue group)
#   - Slight variations in the X and Y axes might represent steady behaviours
#       like carrying it on the pocket (groups green and yellow)
#   - The purple group has a lot of variation in Component2 (related to Z axis).
#       This behaviour might be related to:
#           + Holding phone on hands (using the phone)
#           + Phone falling (outliers)



df['kmeans_group'] = km.labels_

res = df.groupby(('kmeans_group')).mean()
res.plot(kind='bar', legend=True)
plt.show()
res.to_excel('KMeans.xlsx')
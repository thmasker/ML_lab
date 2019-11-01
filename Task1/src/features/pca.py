import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os


FEATURES_FILE = os.path.join('.', 'data', 'processed', 'T2_features.csv')


print(f"Loading extracted features from {FEATURES_FILE}")
df = pd.read_csv(FEATURES_FILE)
df = df.dropna()
print(df.shape)


##  Normalization
# TODO: MinMaxScaler provides better explained variance ratio when using PCA
scaler = preprocessing.StandardScaler()
df_scaled = scaler.fit_transform(df)


##  PCA
estimator = PCA(n_components = 2)
X_pca = estimator.fit_transform(df_scaled)
print("[ Explained variance ratio ]")
print(estimator.explained_variance_ratio_)
print(sum(estimator.explained_variance_ratio_))
# Interpretation:
#   We got a total explained variance of around 78-79%, which is higher
#   than 75%, so these two components are representative


print("[ Relation between PCA components and features ]")
print(pd.DataFrame(np.matrix.transpose(estimator.components_), columns=['PC-1', 'PC-2'], index=df.columns))
# Interpretation:
#   - Component1 includes mainly contributions from X and Y axes
#   - Component2 includes mainly contributions from Z axis


##  Plot
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.xlabel("PC-1")
plt.ylabel("PC-2")
plt.grid(True)
plt.show()
plt.tight_layout()
# Interpretation:
#   - Since Component2 represents mostly Z axes and there's an outlier,
#       the phone might have fallen
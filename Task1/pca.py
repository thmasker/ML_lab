

import pandas as pd


df = pd.read_csv("T2_features.csv")
df = df.dropna()


from sklearn import preprocessing 


scaler = preprocessing.StandardScaler()
df_scaled = scaler.fit_transform(df)

from sklearn.decomposition import PCA

estimator = PCA(n_components = 2)
X_pca = estimator.fit_transform(df_scaled)
print(estimator.explained_variance_ratio_)
print(sum(estimator.explained_variance_ratio_))


import matplotlib.pyplot as plt
import numpy
numbers = numpy.arange(len(X_pca))
fig, ax = plt.subplots()
for i in range(len(X_pca)):
    plt.plot(X_pca[i][0], X_pca[i][1]) 
#for i in range(len(X_pca)):
#    plt.text(X_pca[i][0], X_pca[i][1], numbers[i]) 
# be careful with xlim and ylim
#plt.xlim(-1, 4)
#plt.ylim(-0.2, 1)
ax.grid(True)
fig.tight_layout()
plt.show()
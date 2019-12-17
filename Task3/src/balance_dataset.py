import os
import pandas as pd
import numpy as np

TRAIN_DATASET = os.path.join('.', 'data', 'processed', 'train.csv')
METHOD = ["downsampling", "kmeans"][0]
FEATURES = [
	# 'UserID',
	# 'UUID',
	# 'Version',
	# 'TimeStemp',
	'GyroscopeStat_x_MEAN',
	'GyroscopeStat_z_MEAN',
	'GyroscopeStat_COV_z_x',
	'GyroscopeStat_COV_z_y',
	'MagneticField_x_MEAN',
	'MagneticField_z_MEAN',
	'MagneticField_COV_z_x',
	'MagneticField_COV_z_y',
	'Pressure_MEAN',
	'LinearAcceleration_COV_z_x',
	'LinearAcceleration_COV_z_y',
	'LinearAcceleration_x_MEAN',
	'LinearAcceleration_z_MEAN',
	# 'attack'
	]


df = pd.read_csv(TRAIN_DATASET)

df_attack = df[df['attack'] == 1]
df_noattack = df[df['attack'] == 0]


if METHOD == "kmeans":
    ## Normalization
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler()
    noattack_scaled = scaler.fit_transform(df_noattack[FEATURES])

    ## K-MEANS PARAMETRIZATION
    init = 'random' # initialization method 
    iterations = 10
    max_iter = 300 
    tol = 1e-04

    from sklearn.cluster import KMeans
    from sklearn import metrics
    '''krange = range(140, 145)

    distortions = []
    silhouettes = []

    for k in krange:
        print("Testing for", k, "clusters")
        km = KMeans(k, init, n_init = iterations, max_iter= max_iter, tol = tol)
        labels = km.fit_predict(noattack_scaled)
        distortions.append(km.inertia_)
        silhouettes.append(metrics.silhouette_score(noattack_scaled, labels))


    import matplotlib.pyplot as plt

    plt.plot(krange, distortions, marker='o')
    plt.xlabel('K')
    plt.ylabel('Distortion')
    plt.show()

    plt.plot(krange, silhouettes , marker='o')
    plt.xlabel('K')
    plt.ylabel('Silhouette')
    plt.show()
    '''

    k = 141

    km = KMeans(k, init, n_init = iterations, max_iter= max_iter, tol = tol)

    km.fit_predict(noattack_scaled)

    import pandas as pd
    df_noattack = pd.DataFrame(data=km.cluster_centers_, columns=FEATURES)
    df_noattack['attack'] = 0

    pd.concat([df_attack[FEATURES + ['attack']], df_noattack], ignore_index=True).to_csv(os.path.join('.', 'data', 'processed', 'train_balanced.csv'), index=False)


elif METHOD == "downsampling":
    print(len(df_noattack))

    #noattack_target_count = len(df_noattack)//2
    noattack_target_count = 200

    drop_indices = np.random.choice(df_noattack.index, len(df_noattack) - noattack_target_count, replace=False)
    df_noattack = df.drop(drop_indices)

    print(len(df_noattack))


    pd.concat([df_attack, df_noattack], ignore_index=True).to_csv(os.path.join('.', 'data', 'processed', 'train_balanced.csv'), index=False)
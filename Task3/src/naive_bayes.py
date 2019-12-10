import os

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# pd.set_option('display.width', None)
# pd.set_option('display.max_columns', None)

df_train = pd.read_csv(os.path.join('..', 'data', 'task3_train.csv'))
df_test = pd.read_csv(os.path.join('..', 'data', 'task3_train.csv'))

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

X_train = df_train[FEATURES]
X_test = df_test[FEATURES]

scaler = preprocessing.StandardScaler().fit(X_train)
X_scaled = scaler.transform(X_train)

scaler = preprocessing.StandardScaler().fit(X_test)
X_test_scaled = scaler.transform(X_test)

y_train = df_train['attack']
y_round = [round(a, 0) for a in y_train]

y_test = df_test['attack']
y_test_round = [round(a, 0) for a in y_test]

model = GaussianNB()
model.fit(X_scaled, y_round)

y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test_round, y_pred)
print("Accuracy", acc)

xx = np.stack(i for i in range(len(y_test_round)))
plt.scatter(xx, y_test_round, c='r', label='data')
plt.plot(xx, y_pred, c='g', label='prediction')
plt.axis('tight')
plt.legend()
plt.title('Gaussian NaiveBayes')
plt.show()

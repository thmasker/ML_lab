import os
import pandas as pd
import random

'''
Training data --> 80 % attacks + 80 % not attacks
Test data --> 20 % attacks + 20 % not attacks
'''
THRESHOLD = 0.8

df = pd.read_csv(os.path.join('.', 'data', 'task3_dataset.csv'))
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

FEATURES = [
	'UserID',
	'UUID',
	'Version',
	'TimeStemp',
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
	'attack']

attack_df = df[df['attack'] == 1]
noattack_df = df[df['attack'] == 0]


train = []
test = []

for i, row in df.iterrows():
	if random.random() < THRESHOLD:
		train.append(row)
	else:
		test.append(row)


df_train = pd.DataFrame(data=train, columns=FEATURES)
df_test = pd.DataFrame(data=test, columns=FEATURES)

df_train[FEATURES].to_csv(os.path.join('.', 'data', 'task3_train.csv'), index=False)
df_test[FEATURES].to_csv(os.path.join('.', 'data', 'task3_test.csv'), index=False)

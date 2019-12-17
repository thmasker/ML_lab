import os
import pandas as pd
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
import random
from sklearn.model_selection import train_test_split

'''
Training data --> 80 % attacks + 80 % not attacks
Test data --> 20 % attacks + 20 % not attacks
'''
TRAIN = 0.8
RAW_DATA = os.path.join('.', 'data', 'raw', 'dataset.csv')


df = pd.read_csv(RAW_DATA)

print("> Attack")
attack_train, attack_test = train_test_split(df[df['attack'] == 1], train_size=TRAIN, shuffle=True)
print("\tTrain:", len(attack_train))
print("\tTest: ", len(attack_test))

print("> No-Attack")
noattack_train, noattack_test = train_test_split(df[df['attack'] == 0], train_size=TRAIN, shuffle=True)
print("\tTrain:", len(noattack_train))
print("\tTest: ", len(noattack_test))

pd.concat([attack_train, noattack_train], ignore_index=True).to_csv(os.path.join('.', 'data', 'processed', 'train.csv'), index=False)
pd.concat([attack_test, noattack_test], ignore_index=True).to_csv(os.path.join('.', 'data', 'processed', 'test.csv'), index=False)

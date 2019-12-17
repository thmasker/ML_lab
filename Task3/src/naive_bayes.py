import os

import matplotlib.pyplot as plt

import pandas as pd
# pd.set_option('display.width', None)
# pd.set_option('display.max_columns', None)
import numpy as np

from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score

df_train = pd.read_csv(os.path.join('.', 'data', 'processed', 'train.csv'))
df_train = df_train.sample(frac=1).reset_index(drop=True)
df_test = pd.read_csv(os.path.join('.', 'data', 'processed', 'test.csv'))
df_test = df_test.sample(frac=1).reset_index(drop=True)

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

y_train = df_train['attack']
y_test = df_test['attack']

## [ NORMALIZATION ]
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)


## [ NAIVE-BAYES MODEL ]

model = BernoulliNB()
#model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

xx = np.stack(i for i in range(len(y_test)))
plt.scatter(xx, y_test, c='r', label='data')
plt.plot(xx, y_pred, c='g', label='prediction')
plt.axis('tight')
plt.legend()
plt.title('Gaussian NaiveBayes')
plt.show()



from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

plot_confusion_matrix(y_test, y_pred , classes = unique_labels(y_test, y_pred), normalize=True,
                      title='Normalized confusion matrix')

plt.show()
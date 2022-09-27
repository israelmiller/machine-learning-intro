#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import tensorflow as tf

#THis dataset is from the UCI Machine Learning Repository
#Dataset: https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope 

#Read in the data
cols = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
df = pd.read_csv('data/magic04.data', names=cols)

#Check the dataframe
df
df.sample(5)

#convert class column to an integer
df['class'] = (df['class'] == 'g').astype(int)

### Data Exploration
#for label in cols[:-1]:
#    plt.hist(df[df["class"]== 1][label], color='blue', label='gamma', alpha=.7, density=True)
#    plt.hist(df[df["class"]== 0][label], color='red', label='hadron', alpha=.7, density=True)
#    plt.title(label)
#    plt.ylabel('Probability')
#    plt.xlabel(label)
#    plt.legend()
#    plt.show()
####

#Split the data into training, validation, and testing
train, valid, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

def scale_dataset(dataframe, oversample = False):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scalar = StandardScaler()
    x = scalar.fit_transform(x)
    
    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)

    data = np.hstack((x, np.reshape(y, (-1, 1))))

    return data, x, y

#Scale the datasets 
train, x_train, y_train = scale_dataset(train, oversample=True)
valid, x_valid, y_valid = scale_dataset(valid, oversample=False)
test, x_test, y_test = scale_dataset(test, oversample=False)

#K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=2)
knn_model.fit(x_train, y_train)

y_pred = knn_model.predict(x_test)

print(classification_report(y_test, y_pred))

#Naive Bayes

nb_model = GaussianNB()
nb_model = nb_model.fit(x_train, y_train)

y_pred = nb_model.predict(x_test)


print(classification_report(y_test, y_pred))

#Logistic regression

lr_model = LogisticRegression()
lr_model = lr_model.fit(x_train, y_train)

y_pred = lr_model.predict(x_test)

print(classification_report(y_test, y_pred))

#Support Vector Machine

svm_model = SVC()
svm_model = svm_model.fit(x_train, y_train)

y_pred = svm_model.predict(x_test)

print(classification_report(y_test, y_pred))

## Neural Networks

inmport tensorflow as tf





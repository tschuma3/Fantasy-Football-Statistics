import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

#Importing the dataset
dataset = pd.read_csv(r'Fantasy Football Dataset - Sheet1.csv')
X = dataset.iloc[:, :].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

#Label Encoding
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
X[:, 2] = le.fit_transform(X[:, 2])
X[:, 3] = le.fit_transform(X[:, 3])

print(X)
print(y)

#One Hot Encoding
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = ct.fit_transform(X)

print(X)
print(y)

#Splitting the dataset into a training and testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Feature Scaling
sc = StandardScaler(with_mean=False)
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Initializing the ANN
ann = Sequential()

#Adding the input layer and the first hidden layer
ann.add(Dense(units=17, activation='relu'))

#Adding the second hidden layer
ann.add(Dense(units=10, activation='relu'))

#Adding the output layer
ann.add(Dense(units=1, activation='sigmoid'))

#Compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Training the ANN
ann.fit(X_train, y_train, batch_size=32, epochs=100)

#Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("CM: " + str(cm))
print("Accuracy Score: " + str(accuracy_score(y_test, y_pred)))
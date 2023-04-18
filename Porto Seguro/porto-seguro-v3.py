#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 08:55:38 2018

@author: rafaelleite
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
    
# Importing the datasets
dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')
dataset = pd.concat([dataset_train, dataset_test])

""" In the train and test data, features that belong to similar groupings are tagged 
as such in the feature names (e.g., ind, reg, car, calc). In addition, feature names 
include the postfix bin to indicate binary features and cat to indicate categorical features. 
Features without these designations are either continuous or ordinal. 
Values of -1 indicate that the feature was missing from the observation. 
The target columns signifies whether or not a claim was filed for that policy holder."""

# Replace -1 values with nan on dataset
dataset = dataset.replace(to_replace = -1, value = np.NaN)

# split dataset in X, y
X = dataset.iloc[:, 1:-1].values
y = pd.DataFrame(dataset.iloc[:, 58]).values

# Filling the null values
#dataset.info()
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)
X = pd.DataFrame(data = X, columns = dataset.iloc[:, 1:-1].columns)
#X.info()
#X.dtypes

# setting dtype to category for categorical data
cols = [col for col in X.columns if '_cat' in col]
for col in X[cols]:
    X[col] = X[col].astype('category')
#X.dtypes

"""# check how many columns we have and delete irrelevant data
ps_car_11_cat = pd.DataFrame(data = X['ps_car_11_cat'].values.ravel(), columns = ['values'])
ps_car_11_cat.apply(pd.value_counts)
cols = [col for col in X.columns if 'ps_car_11_cat' in col]
X = X.drop(columns=cols)"""

# for the categorical features, we will add dummies
X = pd.get_dummies(X, drop_first = True)

# splitting into test and train datasets
X_train = X.iloc[:595212, :].values
X_test = X.iloc[595212:, :].values
y_train = y[:595212, :]
y_test = y[595212:, :]

# Apply the random under-sampling for X_train
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42, ratio = 0.95)
X_train_resampled, y_train_resampled = rus.fit_sample(X_train, y_train.reshape(-1,1).ravel())

# feature scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_resampled = sc.fit_transform(X_train_resampled)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


# Feature Selection
# Backward Eliminiation

# Insert B Intercept
X_train_resampled = np.append(arr = np.ones((len(X_train_resampled), 1)).astype(int), values = X_train_resampled, axis = 1)

# Call Ordinary Least Square and Clean irrelevant data from X_train (except dummies)
import statsmodels.formula.api as sm
regressorOLS = sm.OLS(y_train_resampled, X_train_resampled).fit()
regressorOLS.summary()

max_column = len(X_train_resampled[1])

for k in range(0,100):
    for i in range (0 , max_column):
        regressorOLS = sm.OLS(y_train_resampled, X_train_resampled).fit()
        pvalue = []
    
        if i < (max_column -1):
            for j in range (1,(max_column - i)):
                pvalue.append(regressorOLS.pvalues[j])
    
            if (max(pvalue) > 0.05):
                index = np.argmax(pvalue) + 1
                X_train_resampled = np.delete(X_train_resampled, index, 1) 
                X_train = np.delete(X_train, (index - 1), 1) 
                X_test = np.delete(X_test, (index - 1), 1) 
                i = 0
                max_column = max_column - 1
            else:
                break
       
regressorOLS = sm.OLS(y_train_resampled, X_train_resampled).fit()
regressorOLS.summary()
regressorOLS.pvalues

X_train_resampled = np.delete(X_train_resampled, 0, 1) 

# Splitting Train dataset in N random sample datasets (R% ratio) for training N different ANN's
from random import randrange, uniform
number_of_ann_s = 50
sample_ratio = 1
list_of_dataframes = []

for i in range (0,number_of_ann_s):
    list_of_dataframes.append(pd.DataFrame())

X_train_resampled_fraction = []
y_train_resampled_fraction = []

X_train_resampled_fraction = list(list_of_dataframes)
y_train_resampled_fraction = list(list_of_dataframes)

X_train_resampled_y_train_resampled = X_train_resampled
X_train_resampled_y_train_resampled = np.concatenate((X_train_resampled, y_train_resampled[:,None]), axis = 1)
X_train_resampled_y_train_resampled = pd.DataFrame(data = X_train_resampled_y_train_resampled)

for i in range (0,number_of_ann_s):
    X_train_resampled_fraction[i] = pd.DataFrame(data = X_train_resampled_y_train_resampled.sample(frac = uniform(0.5,1) * sample_ratio))
    y_train_resampled_fraction[i] = X_train_resampled_fraction[i].iloc[:,44].values
    X_train_resampled_fraction[i] = X_train_resampled_fraction[i].iloc[:,:-1]


# Part 2 - Artificial Neural Network
# Importing the Keras libraries and packages

import keras
from keras.models import Sequential
from keras.layers import Dense

y_pred_train = []
y_pred_train = list(list_of_dataframes )
y_pred_test = []
y_pred_test = list(list_of_dataframes)


for i in range (0,number_of_ann_s):    

    # Initialising the ANN
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense((5 * randrange(1,3)), kernel_initializer = 'uniform', activation = 'relu', input_dim = len(X_train_resampled[1])))
    
    # Adding the output layer
    classifier.add(Dense(1, init = 'uniform', activation = 'sigmoid'))
    
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
    
    # Fitting the ANN to the Training set
    classifier.fit(X_train_resampled_fraction[i], y_train_resampled_fraction[i], batch_size = 32, epochs = 110)

    # evaluate the model
    #scores = classifier.evaluate(X_train, y_train)
    #print("\n%s: %.2f%%, ann:%i" % (classifier.metrics_names[1], scores[1]*100, i))
    print("i:%s", i)
    
    # Make prediction
    y_pred_test[i] = classifier.predict(X_test)
        

   

# Predicting the Test set results
y_pred = pd.DataFrame(columns = ['id', 'target'])
y_pred['id'] = dataset.iloc[595212:, 0].values
y_pred['target'] = np.mean(y_pred_test,axis=0)
y_pred['target'].min()
y_pred['target'].max()


#Part 3 - Generating Submission File
import zipfile
try:
    import zlib
    compression = zipfile.ZIP_DEFLATED
except:
    compression = zipfile.ZIP_STORED

modes = { zipfile.ZIP_DEFLATED: 'deflated',
          zipfile.ZIP_STORED:   'stored',
          }
y_pred.to_csv('porto_seguro_submission.csv', index = False)
zf = zipfile.ZipFile('porto_seguro_submission.zip', mode='w')
try:
    zf.write('porto_seguro_submission.csv', compress_type=compression)
finally:
    zf.close()


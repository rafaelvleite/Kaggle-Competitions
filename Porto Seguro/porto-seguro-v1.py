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
X = dataset.iloc[:, 1:-1]
y = pd.DataFrame(dataset.iloc[:, 58])

# Filling the null values
dataset.info()
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

# check how many columns we have and delete irrelevant data
ps_car_11_cat = pd.DataFrame(data = X['ps_car_11_cat'], columns = ['values'])
ps_car_11_cat.apply(pd.value_counts)
cols = [col for col in X.columns if 'ps_car_11_cat' in col]
X = X.drop(columns=cols)

# for the categorical features, we will add dummies
X = pd.get_dummies(X, drop_first = True)

# splitting into test and train datasets
X_train = X.iloc[:595212, :]
X_test = X.iloc[595212:, :]
y_train = y.iloc[:595212, :]
y_test = y.iloc[595212:, :]

# Apply the random under-sampling for X_train
from imblearn.under_sampling import RandomUnderSampler
imbalance_checker = y.apply(pd.value_counts)
under_ratio = 1 - imbalance_checker.target[1]/imbalance_checker.target[0]
rus = RandomUnderSampler(ratio = under_ratio, random_state=42)
X_train_resampled, y_train_resampled = rus.fit_sample(X_train, y_train.values.ravel())
X_train_resampled = pd.DataFrame(X_train_resampled, columns = X_train.columns)
y_train_resampled = pd.DataFrame(y_train_resampled, columns = y_train.columns)
imbalance_checker = y_train_resampled.apply(pd.value_counts)

"""# Apply the random over-sampling for X_train
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_sample(X_train, y_train.values.ravel())
X_train_resampled = pd.DataFrame(X_train_resampled, columns = X_train.columns)
y_train_resampled = pd.DataFrame(y_train_resampled, columns = y_train.columns)
imbalance_checker = y_train_resampled.apply(pd.value_counts)"""

# for the continuous or ordinal features, we will use feature scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
cols = [col for col in X_train_resampled.columns if not '_bin' in col and not 'cat' in col]
X_train_resampled[cols] = sc.fit_transform(X_train_resampled[cols])
X_train[cols] = sc.transform(X_train[cols])
X_test[cols] = sc.transform(X_test[cols])


# Backward Eliminiation

# Insert B Intercept
X_train_resampled['constant'] = 1
X_train_resampled = X_train_resampled[['constant'] + X_train_resampled.columns[:-1].tolist()]

# Call Ordinary Least Square and Clean irrelevant data from X_train (except dummies)
import statsmodels.formula.api as sm
xelimination = X_train_resampled
regressorOLS = sm.OLS(y_train_resampled.values.ravel(), xelimination).fit()
regressorOLS.summary()

deleted_columns = []
deleted_pvalues = []

max_column = len(X_train_resampled.columns)

for k in range(0,10):
    for i in range (0 , max_column):
        xelimination = X_train_resampled
        regressorOLS = sm.OLS(y_train_resampled.values.ravel(), xelimination).fit()
    
        pvalue = []
    
        if i < (max_column -1):
            for j in range (1,(max_column - i)):
                pvalue.append(regressorOLS.pvalues[j])
    
            if (max(pvalue) > 0.05):
                index = np.argmax(pvalue) + 1
                if (X_train_resampled.columns[index] != 'constant'):
                    deleted_columns.append(X_train_resampled.columns[index])
                    deleted_pvalues.append(max(pvalue))
                    X_train_resampled.drop(X_train_resampled.columns[index], axis=1, inplace=True)
                    i = 0
                    max_column = max_column - 1
                else:
                    print(index)
                    break
            else:
                break
        

regressorOLS.summary()
regressorOLS.pvalues
X_train_resampled.drop(['constant'], axis=1, inplace=True)
X_train.drop(deleted_columns, axis = 1, inplace = True)
X_test.drop(deleted_columns, axis = 1, inplace = True)

    
# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)
model.fit(X_train_resampled,y_train_resampled.values.ravel())
y_pred_ann = []
y_pred_ann = model.predict_proba(X_test)
y_pred_ann = y_pred_ann[:,1]

y_pred = model.predict(X_train)
y_pred = y_pred.reshape(595212)
y_true = y_train.values.ravel().astype('float32') 


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 3000, random_state = 0)
regressor.fit(X_train_resampled,y_train_resampled.values.ravel())
y_pred_ann_2 = regressor.predict_proba(X_test)


try:
    model_json
except NameError:
    # Initialising the ANN
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(5, init = 'uniform', activation = 'relu', input_dim = len(X_train_resampled.columns)))
    
    # Adding the output layer
    classifier.add(Dense(1, init = 'uniform', activation = 'sigmoid'))
    
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
    
    # Fitting the ANN to the Training set
    history = classifier.fit(X_train_resampled, y_train_resampled.values.ravel(), batch_size = 32, epochs = 80)
    classifier_history = pd.DataFrame(history.history)
    plt.figure(); classifier_history['loss'].plot();

    # evaluate the model
    scores = classifier.evaluate(X_train, y_train)
    print("\n%s: %.2f%%" % (classifier.metrics_names[1], scores[1]*100))

    # Simple implementation of the (normalized) gini score in numpy
# Fully vectorized, no python loops, zips, etc.
# Significantly (>30x) faster than previous implementions

    y_true = y_train.values.ravel().astype('float32') 
    y_pred = classifier.predict(X_train)
    y_pred = y_pred.reshape(595212)
    
    def Gini(y_true, y_pred):
        # check and get number of samples
        assert y_true.shape == y_pred.shape
        n_samples = y_true.shape[0]
        
        # sort rows on prediction column 
        # (from largest to smallest)
        arr = np.array([y_true, y_pred]).transpose()
        true_order = arr[arr[:,0].argsort()][::-1,0]
        pred_order = arr[arr[:,1].argsort()][::-1,0]
        
        # get Lorenz curves
        L_true = np.cumsum(true_order) / np.sum(true_order)
        L_pred = np.cumsum(pred_order) / np.sum(pred_order)
        L_ones = np.linspace(1/n_samples, 1, n_samples)
        
        # get Gini coefficients (area between curves)
        G_true = np.sum(L_ones - L_true)
        G_pred = np.sum(L_ones - L_pred)
        
        # normalize to true Gini coefficient
        return G_pred/G_true

    Gini(y_true, y_pred)
    
    
    
    
    # Saving the Trained Model
    # serialize model to JSON
    model_json = classifier.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    classifier.save_weights("model.h5")
    print("Saved model to disk")
    

else:
     # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    classifier = model_from_json(loaded_model_json)
    # load weights into new model
    classifier.load_weights("model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = classifier.evaluate(X_train, y_train, verbose=0)
    print("%s: %.2f%%" % (classifier.metrics_names[1], score[1]*100))

   

# Predicting the Test set results
y_pred_ann = classifier.predict(X_test)
y_pred = pd.DataFrame(columns = ['id', 'target'])
y_pred['id'] = dataset.iloc[595212:, 0].values
y_pred['target'] = y_pred_ann_2
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


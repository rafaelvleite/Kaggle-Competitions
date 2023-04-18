# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Part 1 - Preprocessing the data

# Importing the Libraries
import numpy as np
import pandas as pd


# Importing the test dataset
dataset_test = pd.read_csv('test.csv')

# Importing the training dataset
dataset_train = pd.read_csv('train.csv')

# Cleaning the data
dataset_test.drop(['Name', 'Ticket', 'Cabin' ], axis = 1, inplace = True)
dataset_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin' ], axis = 1, inplace = True)
dataset_train['Age'].fillna(0, inplace=True)
dataset_train.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

#Enconding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_sex = LabelEncoder()
dataset_test.iloc[:, 2] = labelencoder_sex.fit_transform(dataset_test.iloc[:, 2])
dataset_train.iloc[:, 2] = labelencoder_sex.fit_transform(dataset_train.iloc[:, 2])

labelencoder_embarked = LabelEncoder()
dataset_test.iloc[:, 7] = labelencoder_embarked.fit_transform(dataset_test.iloc[:, 7])
dataset_train.iloc[:, 7] = labelencoder_embarked.fit_transform(dataset_train.iloc[:, 7])


# Estimating and updating Age and Fare for the null values on train and test dataset

# There are two genders and three passenger classes in this dataset. 
# So we create a 2 by 3 matrix to store the median values.
 
# Create a 2 by 3 matrix of zeroes
median_ages = np.zeros((2,3))
median_fares = np.zeros((2,3))
 
# For each cell in the 2 by 3 matrix
for i in range(0,2):
    for j in range(0,3):
 
    	# Set the value of the cell to be the median of all `Age` values
    	# matching the criterion 'Corresponding gender and Pclass',
    	# leaving out all NaN values
        median_ages[i,j] = dataset_train[ (dataset_train['Sex'] == i) & \
                               (dataset_train['Pclass'] == j+1)]['Age'].dropna().median()
        median_fares[i,j] = dataset_train[ (dataset_train['Sex'] == i) & \
                               (dataset_train['Pclass'] == j+1)]['Fare'].dropna().median()

    
# Create new columns AgeFill and FareFill to put values into. 
# This retains the state of the original data.
dataset_test['AgeFill'] = dataset_test['Age']
dataset_test[ dataset_test['Age'].isnull()][['Age', 'AgeFill', 'Sex', 'Pclass']].head(10)
dataset_test['FareFill'] = dataset_test['Fare']
dataset_test[ dataset_test['Fare'].isnull()][['Fare', 'AgeFill', 'Sex', 'Pclass']].head(11)

dataset_train['AgeFill'] = dataset_train['Age']
dataset_train[ dataset_train['Age'].isnull()][['Age', 'AgeFill', 'Sex', 'Pclass']].head(8)
dataset_train['FareFill'] = dataset_train['Fare']
dataset_train[ dataset_train['Fare'].isnull()][['Fare', 'AgeFill', 'Sex', 'Pclass']].head(9)

dataset_train['Age'] = dataset_train['Age'].replace(0, np.nan)
 
# Put our estimates into NaN rows of new columns AgeFill and FareFill.
# df.loc is a purely label-location based indexer for selection by label.
 
for i in range(0, 2):
    for j in range(0, 3):
 
    	# Locate all cells in dataframe where `Sex` == i, `Pclass` == j+1
    	# and `Age` == null and 'Fare' == null. 
    	# Replace them with the corresponding estimate from the matrix.
        dataset_test.loc[ (dataset_test.Age.isnull()) & (dataset_test.Sex == i) & (dataset_test.Pclass == j+1),\
                 'AgeFill'] = median_ages[i,j]	
        dataset_test.loc[ (dataset_test.Fare.isnull()) & (dataset_test.Sex == i) & (dataset_test.Pclass == j+1),\
                 'FareFill'] = median_fares[i,j]	
        
        dataset_train.loc[ (dataset_train.Age.isnull()) & (dataset_train.Sex == i) & (dataset_train.Pclass == j+1),\
                 'AgeFill'] = median_ages[i,j]	
        dataset_train.loc[ (dataset_train.Fare.isnull()) & (dataset_train.Sex == i) & (dataset_train.Pclass == j+1),\
                 'FareFill'] = median_fares[i,j]	



# Create a feature that records whether the Age was originally missing
dataset_test['AgeIsNull'] = pd.isnull(dataset_test['Age']).astype(int)
dataset_test['FareIsNull'] = pd.isnull(dataset_test['Fare']).astype(int)
dataset_test.head()

dataset_train['AgeIsNull'] = pd.isnull(dataset_train['Age']).astype(int)
dataset_train['FareIsNull'] = pd.isnull(dataset_train['Fare']).astype(int)
dataset_train.head()


# Now we remove the null values from the test dataset and we clean the columns Age and Fare
dataset_test.drop(['Age', 'Fare' ], axis = 1, inplace = True)
dataset_test.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

dataset_train.drop(['Age', 'Fare' ], axis = 1, inplace = True)
dataset_train.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

# The 'Embarked' column has 3 types of data, with no Hierarchy, so we need to
# create 3 dummy variables and exclude 1 one of them to avoid the dummy variable trap
onehotencoder = OneHotEncoder(categorical_features = [5])
dataset_test = onehotencoder.fit_transform(dataset_test).toarray()
dataset_test = dataset_test[:, 1:]
dataset_train = onehotencoder.fit_transform(dataset_train).toarray()
dataset_train = dataset_train[:, 1:]

# Splitting the datasets into the input and output
X_train = dataset_train[:, [0,1,3,4,5,6,7,8]]
y_train = dataset_train[:, [2]]
X_test = dataset_test[:, [0,1,3,4,5,6,7,8]]

#getting the PassIndex from the Test dataset
pass_index = dataset_test[:,2]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer with Droput
classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))

# Adding the second hidden layer
classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 32, epochs = 50)






# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# We now consider any prediction >0.5 as 1 and <=0.5 as 0
y_pred = np.round_(y_pred,0)





# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))
    classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = Sequential()
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 64, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout regularization to reduce overfitting if needed


# Tunning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer, units, loss, initializer, metrics, activation):
    classifier = Sequential()
    classifier.add(Dense(units = units, kernel_initializer = initializer, activation = activation, input_dim = 8))
    classifier.add(Dense(units = units, kernel_initializer = initializer, activation = activation))
    classifier.add(Dense(units = 1, kernel_initializer = initializer, activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = loss, metrics = [metrics])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [32], 
              'epochs': [50, 100, 200],
              'optimizer': ['rmsprop'],
              'units': [50],
              'loss': ['binary_crossentropy'],
              'initializer': ['uniform'],
              'metrics': ['accuracy'],
              'activation': ['relu']
              }
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_




#Part 4 - Generating Study and Submission Files

# Let's get the original Test Dataset again and include the predictions
dataset_test_submit = pd.read_csv('test.csv')
dataset_test_submit['Survived Prediction'] = y_pred

# Finally, we generate the output csv file            
dataset_test_submit.to_csv('titanic_submission.csv', sep='\t', encoding='utf-8')
only_final_values = pd.read_csv('test.csv')
only_final_values['Survived'] = y_pred
only_final_values = only_final_values.iloc[: , [0,11]]
only_final_values['Survived'] = only_final_values['Survived'].astype(np.int64)
only_final_values.to_csv('titanic_submission_final.csv', index = False)

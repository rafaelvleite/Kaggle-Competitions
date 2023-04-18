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

# Joining both datasets
df1 = pd.DataFrame(dataset_train)
df2 = pd.DataFrame(dataset_test)
dataset_joined = [df1, df2]
dataset_joined = pd.concat(dataset_joined)


# Cleaning the data

# removing everything from the names before the comma and after the dot, so we have only titles
dataset_joined['Name'] = dataset_joined['Name'].str.split(',').str[1]
dataset_joined['Name'] = dataset_joined['Name'].str.split('.').str[0]

# Categorizing the 'Name' data
names = dataset_joined['Name']
for item in names:
    if (item == ' Mr' or item == ' Master' or item == ' Rev' or item == ' Don' or item == ' Dr' or item == ' Major' or item == ' Sir' or item == ' Col'):
        names.replace(item, 1, inplace = True)
    else:
        names.replace(item, 0, inplace = True)
dataset_joined['Name'] = names
        

# droping irrelevant data
dataset_joined.drop(['Ticket', 'Cabin' ], axis = 1, inplace = True)

# Filling nan values Embarked with 'S', the most relevant data
dataset_joined['Embarked'].fillna('S', inplace=True)

dataset_joined.info()



#Enconding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_sex = LabelEncoder()
dataset_joined['Sex'] = labelencoder_sex.fit_transform(dataset_joined['Sex'])

labelencoder_embarked = LabelEncoder()
dataset_joined['Embarked'] = labelencoder_embarked.fit_transform(dataset_joined['Embarked'])


# Estimating and updating Age and Fare for the null values on dataset


# Getting average Age and Fare
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
        median_ages[i,j] = dataset_joined[ (dataset_joined['Sex'] == i) & \
                               (dataset_joined['Pclass'] == j+1)]['Age'].dropna().median()
        median_fares[i,j] = dataset_joined[ (dataset_joined['Sex'] == i) & \
                               (dataset_joined['Pclass'] == j+1)]['Fare'].dropna().median()

# Create new columns AgeFill and FareFill to put values into. 
# This retains the state of the original data.
dataset_joined['AgeFill'] = dataset_joined['Age']
dataset_joined[ dataset_joined['Age'].isnull()][['Age', 'AgeFill', 'Sex', 'Pclass']].head(10)
dataset_joined['FareFill'] = dataset_joined['Fare']
dataset_joined[ dataset_joined['Fare'].isnull()][['Fare', 'AgeFill', 'Sex', 'Pclass']].head(11)

# Put our estimates into NaN rows of new columns AgeFill and FareFill.
# df.loc is a purely label-location based indexer for selection by label.
 
for i in range(0, 2):
    for j in range(0, 3):
 
    	# Locate all cells in dataframe where `Sex` == i, `Pclass` == j+1
    	# and `Age` == null and 'Fare' == null. 
    	# Replace them with the corresponding estimate from the matrix.
        dataset_joined.loc[ (dataset_joined.Age.isnull()) & (dataset_joined.Sex == i) & (dataset_joined.Pclass == j+1),\
                 'AgeFill'] = median_ages[i,j]	
        dataset_joined.loc[ (dataset_joined.Fare.isnull()) & (dataset_joined.Sex == i) & (dataset_joined.Pclass == j+1),\
                 'FareFill'] = median_fares[i,j]	
        

# Create a feature that records whether the Age was originally missing
dataset_joined['AgeIsNull'] = pd.isnull(dataset_joined['Age']).astype(int)
dataset_joined['FareIsNull'] = pd.isnull(dataset_joined['Fare']).astype(int)
dataset_joined.head()


# Now we remove the null values from the test dataset and we clean the columns Age and Fare
dataset_joined.drop(['Age', 'Fare' ], axis = 1, inplace = True)

# Filling no Survived data with -1
dataset_joined['Survived'].fillna(-1, inplace=True)

dataset_joined.info()

headers = list(dataset_joined)
headers_adicionais = list(['Embarked_2'])
headers = headers_adicionais + headers


# The 'Embarked' column has 3 types of data, with no value Hierarchy, so we need to
# create 3 dummy variables and exclude 1 one of them to avoid the dummy variable trap
onehotencoder = OneHotEncoder(categorical_features = [0])
dataset_joined = onehotencoder.fit_transform(dataset_joined).toarray()
dataset_joined = dataset_joined[:, 1:]
dataset_joined = pd.DataFrame(dataset_joined)

dataset_joined.columns = headers
dataset_joined.info()




# Splitting the dataset into Train and Test
dataset_train_revised = dataset_joined.iloc[:891, :]
dataset_test_revised = dataset_joined.iloc[891:, :]

# Splitting the dataset into the input and output
X_train = dataset_train_revised.iloc[:, [0,1,2,3,5,7,9,10]]
y_train = dataset_train_revised.iloc[:, [8]]
X_test = dataset_test_revised.iloc[:, [0,1,2,3,5,7,9,10]]

#getting the PassIndex for the Submission dataset
pass_index = dataset_joined.iloc[891:,6]

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
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer with Dropout to reduce variance of accuracy
classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))
classifier.add(Dropout(0.3))

# Adding the second hidden layer with Dropout
classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.add(Dropout(0.3))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 25, epochs = 100)






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
from keras.layers import Dropout
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = Sequential()
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 32, epochs = 100)
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
    classifier.add(Dense(units = units, kernel_initializer = initializer, activation = activation, input_dim = 10))
    classifier.add(Dense(units = units, kernel_initializer = initializer, activation = 'sigmoid'))
    classifier.add(Dense(units = 1, kernel_initializer = initializer, activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = loss, metrics = [metrics])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25], 
              'epochs': [100],
              'optimizer': ['rmsprop', 'adam'],
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
only_final_values.to_csv('titanic_submission_final_3.csv', index = False)

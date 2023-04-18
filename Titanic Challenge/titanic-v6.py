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

# droping irrelevant data
dataset_joined.drop(['Ticket', 'Cabin' ], axis = 1, inplace = True)

# Filling nan values Embarked with 'S', the most relevant data
dataset_joined['Embarked'].fillna('S', inplace=True)

dataset_joined.info()



# Enconding Categorical Data

# Categorizing the 'Name' data by treatment hierarquy and gender segregation
names = dataset_joined['Name'].copy()
for item in names:
    if (item == ' Mr'):
        names.replace(item, 1, inplace = True)
    elif (item == ' Miss' or item == ' Mrs'):
        names.replace(item, 0, inplace = True)
    elif (item == ' Capt' or item == ' Col' or item == ' Don' or item == ' Dona' or item == ' Dr' or item == ' Jonkheer' or item == ' Lady' or item == ' Major' or item == ' Master' or item == ' Mile' or item == ' Mlle' or item == ' Mme' or item == ' Ms' or item == ' Rev' or item == ' Sir' or item == ' the Countess'):
        names.replace(item, 2, inplace = True)
    
dataset_joined['Name'] = names




from sklearn.preprocessing import LabelEncoder

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
dataset_joined[ dataset_joined['Age'].isnull()][['Age', 'AgeFill', 'Sex', 'Pclass']]
dataset_joined['FareFill'] = dataset_joined['Fare']
dataset_joined[ dataset_joined['Fare'].isnull()][['Fare', 'AgeFill', 'Sex', 'Pclass']]

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



# The 'Embarked', 'Name' and 'Pclass' columns has several types of data, with no value Hierarchy, so we need to
# create dummy variables and exclude 1 one of them to avoid the dummy variable trap

dataset_joined = pd.get_dummies(dataset_joined, columns=['Embarked', 'Name', 'Pclass'], drop_first=True)


# Splitting the dataset into Train and Test
dataset_train_revised = dataset_joined.iloc[:891, :]
dataset_test_revised = dataset_joined.iloc[891:, :]

# Splitting the dataset into the input and output
X_train = dataset_train_revised.iloc[:,[0,2,3,5,6,9,10,11,12,13,14]]
y_train = dataset_train_revised.iloc[:, [4]]
X_test = dataset_test_revised.iloc[:,[0,2,3,5,6,9,10,11,12,13,14]]

#getting the PassIndex for the Submission dataset
pass_index = dataset_joined.iloc[891:,1]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)








# Part 2 - Grid Search to find best hyperparameters


# Tunning Batch size and number of epochs
import numpy
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

# Function to create model, required for KerasClassifier
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(24, input_dim = 11, activation = 'relu'))
    classifier.add(Dense(1, activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# create model
classifier = KerasClassifier(build_fn = build_classifier)

# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# best parameters are batch_size = 10 and epochs = 50
    
    
    
    
# Tunning Training Optimization Algorithm
import numpy
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

# Function to create model, required for KerasClassifier
def build_classifier(optimizer = 'adam'):
    classifier = Sequential()
    classifier.add(Dense(24, input_dim = 11, activation = 'relu'))
    classifier.add(Dense(1, activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# create model
classifier = KerasClassifier(build_fn = build_classifier, epochs = 50, batch_size = 10)

# define the grid search parameters
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(optimizer=optimizer)
grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# best training optimization is SGD with accuracy 0.832772
    



# Tunning Learning Rate and Momentum
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD

# Function to create model, required for KerasClassifier
def build_classifier(learn_rate=0.01, momentum=0):
    classifier = Sequential()
    classifier.add(Dense(24, input_dim = 11, activation = 'relu'))
    classifier.add(Dense(1, activation = 'sigmoid'))
    optimizer = SGD(lr = learn_rate, momentum = momentum)
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# create model
classifier = KerasClassifier(build_fn = build_classifier, epochs = 50, batch_size = 10)

# define the grid search parameters
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
param_grid = dict(learn_rate=learn_rate, momentum=momentum)
grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# Best: 0.831650 using {'learn_rate': 0.01, 'momentum': 0.9}
    



# Tunning Network Weight Initialization
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifier
def build_classifier(init_mode='uniform'):
    classifier = Sequential()
    classifier.add(Dense(24, input_dim = 11, kernel_initializer=init_mode, activation = 'relu'))
    classifier.add(Dense(1, kernel_initializer=init_mode, activation = 'sigmoid'))
    classifier.compile(optimizer = 'SGD', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# create model
classifier = KerasClassifier(build_fn = build_classifier, epochs = 50, batch_size = 10)

# define the grid search parameters
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
param_grid = dict(init_mode=init_mode)
grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# best network weigth initializtion is glorot_normal




# Tunning Neuron Activation Function
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifier
def build_classifier(activation='relu'):
    classifier = Sequential()
    classifier.add(Dense(24, input_dim = 11, kernel_initializer='glorot_normal', activation = activation))
    classifier.add(Dense(1, kernel_initializer='glorot_normal', activation = 'sigmoid'))
    classifier.compile(optimizer = 'SGD', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# create model
classifier = KerasClassifier(build_fn = build_classifier, epochs = 50, batch_size = 10)

# define the grid search parameters
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
param_grid = dict(activation=activation)
grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# best neuron activation function is 'tanh'





# Tunning Dropout Regularization
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
# Function to create model, required for KerasClassifier
def build_classifier(dropout_rate=0.0, weight_constraint=0):
    classifier = Sequential()
    classifier.add(Dense(24, input_dim = 11, kernel_initializer='glorot_normal', activation = 'tanh', kernel_constraint=maxnorm(weight_constraint)))
    classifier.add(Dropout(dropout_rate))
    classifier.add(Dense(1, kernel_initializer='glorot_normal', activation = 'sigmoid'))
    classifier.compile(optimizer = 'SGD', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# create model
classifier = KerasClassifier(build_fn = build_classifier, epochs = 50, batch_size = 10)

# define the grid search parameters
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# best dropout rate is 0.0 and weight constraint = 5 - So NO Dropout needed!






# Tunning Number of Neurons in the Hidden Layer
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifier
def build_classifier(neurons=1):
    classifier = Sequential()
    classifier.add(Dense(neurons, input_dim = 11, kernel_initializer='glorot_normal', activation = 'tanh'))
    classifier.add(Dense(1, kernel_initializer='glorot_normal', activation = 'sigmoid'))
    classifier.compile(optimizer = 'SGD', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# create model
classifier = KerasClassifier(build_fn = build_classifier, epochs = 50, batch_size = 10)

# define the grid search parameters
neurons = [1, 5, 10, 15, 20, 25, 30]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# best quantity of neurons in hidden layer is 15




# Evaluating the ANN
import numpy
import keras
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
def build_classifier():
    # Initialising the ANN
    classifier = Sequential()
    # Adding the input layer and the first hidden
    classifier.add(Dense(15, activation = 'tanh', input_dim = 11))
    # Adding the output layer
    classifier.add(Dense(1, activation = 'sigmoid'))    
    # Compiling the ANN
    optimizer = SGD(lr = 0.01, momentum = 0.9)
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# create model
classifier = Sequential()

# summarize results
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 50)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()









# Part 3 - Now let's make the ANN!

# Importing the Keras libraries and packages
import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden
classifier.add(Dense(15, activation = 'tanh', input_dim = 11))

# Adding the output layer
classifier.add(Dense(1, activation = 'sigmoid'))

# Compiling the ANN
optimizer = SGD(lr = 0.01, momentum = 0.9)
classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 50)






# Part 4 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# We now consider any prediction >0.5 as 1 and <=0.5 as 0
y_pred = np.round_(y_pred,0)


#Part 5 - Generating Study and Submission Files

# Let's get the original Test Dataset again and include the predictions
dataset_test_submit = pd.read_csv('test.csv')
dataset_test_submit['Survived Prediction'] = y_pred

# Finally, we generate the output csv file            
dataset_test_submit.to_csv('titanic_submission.csv', sep='\t', encoding='utf-8')
only_final_values = pd.read_csv('test.csv')
only_final_values['Survived'] = y_pred
only_final_values = only_final_values.iloc[: , [0,11]]
only_final_values['Survived'] = only_final_values['Survived'].astype(np.int64)
only_final_values.to_csv('titanic_submission_final_6.csv', index = False)

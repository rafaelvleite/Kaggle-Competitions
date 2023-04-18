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
dataset = pd.read_csv('train+test_cleaned.csv')


# Cleaning the data

# Dropping PoolQC'
dataset.drop('PoolQC', axis = 1, inplace = True)

# Filling nan values FireplaceQu
dataset.FireplaceQu.fillna(dataset.FireplaceQu.value_counts().idxmax(), inplace = True)

# Filling nan values MSZoning
dataset.MSZoning.fillna(dataset.MSZoning.value_counts().idxmax(), inplace = True)

# Filling nan values MasVnrType
dataset.MasVnrType.fillna(dataset.MasVnrType.value_counts().idxmax(), inplace = True)

# Filling nan values MasVnrArea
dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(dataset['MasVnrArea'].mean() )

# Filling nan values Exterior2nd
dataset.Exterior2nd.fillna(dataset.Exterior2nd.value_counts().idxmax(), inplace = True)

# Filling nan values BsmtQual
dataset.BsmtQual.fillna(dataset.BsmtQual.value_counts().idxmax(), inplace = True)

# Filling nan values BsmtCond
dataset.BsmtCond.fillna(dataset.BsmtCond.value_counts().idxmax(), inplace = True)

# Filling nan values BsmtExposure
dataset.BsmtExposure.fillna(dataset.BsmtExposure.value_counts().idxmax(), inplace = True)

# Filling nan values BsmtFinType1
dataset.BsmtFinType1.fillna(dataset.BsmtFinType1.value_counts().idxmax(), inplace = True)

# Filling nan values BsmtFinType2
dataset.BsmtFinType2.fillna(dataset.BsmtFinType2.value_counts().idxmax(), inplace = True)

# Filling nan values BsmtFinSF1
dataset['BsmtFinSF1'] = dataset['BsmtFinSF1'].fillna(dataset['BsmtFinSF1'].mean() )

# Filling nan values BsmtFinSF2
dataset['BsmtFinSF2'] = dataset['BsmtFinSF2'].fillna(dataset['BsmtFinSF2'].mean() )

# Filling nan values BsmtUnfSF 
dataset['BsmtUnfSF'] = dataset['BsmtUnfSF'].fillna(dataset['BsmtUnfSF'].mean() )

# Filling nan values TotalBsmtSF 
dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].fillna(dataset['TotalBsmtSF'].mean() )

# Filling nan values KitchenQual 
dataset.KitchenQual.fillna(dataset.KitchenQual.value_counts().idxmax(), inplace = True)

# Filling nan values Functional 
dataset.Functional.fillna(dataset.Functional.value_counts().idxmax(), inplace = True)

# Filling nan values GarageFinish
dataset.GarageFinish.fillna(dataset.GarageFinish.value_counts().idxmax(), inplace = True)

# Filling nan values GarageCars 
dataset['GarageCars'] = dataset['GarageCars'].fillna(dataset['GarageCars'].mean() )

# Filling nan values GarageArea 
dataset['GarageArea'] = dataset['GarageArea'].fillna(dataset['GarageArea'].mean() )

# Filling nan values GarageQual
dataset.GarageQual.fillna(dataset.GarageQual.value_counts().idxmax(), inplace = True)

# Filling nan values GarageCond with 'None'
dataset.GarageCond.fillna(dataset.GarageCond.value_counts().idxmax(), inplace = True)

dataset.info()


# Enconding Categorical Data
from sklearn.preprocessing import LabelEncoder

MSZoning = LabelEncoder()
dataset['MSZoning'] = MSZoning.fit_transform(dataset['MSZoning'])

Street = LabelEncoder()
dataset['Street'] = Street.fit_transform(dataset['Street'])

LandContour = LabelEncoder()
dataset['LandContour'] = LandContour.fit_transform(dataset['LandContour'])

LotConfig = LabelEncoder()
dataset['LotConfig'] = LotConfig.fit_transform(dataset['LotConfig'])

LandSlope = LabelEncoder()
dataset['LandSlope'] = LandSlope.fit_transform(dataset['LandSlope'])

Neighborhood = LabelEncoder()
dataset['Neighborhood'] = Neighborhood.fit_transform(dataset['Neighborhood'])

Condition1 = LabelEncoder()
dataset['Condition1'] = Condition1.fit_transform(dataset['Condition1'])

Condition2 = LabelEncoder()
dataset['Condition2'] = Condition2.fit_transform(dataset['Condition2'])

BldgType = LabelEncoder()
dataset['BldgType'] = BldgType.fit_transform(dataset['BldgType'])

HouseStyle = LabelEncoder()
dataset['HouseStyle'] = HouseStyle.fit_transform(dataset['HouseStyle'])

RoofMatl = LabelEncoder()
dataset['RoofMatl'] = RoofMatl.fit_transform(dataset['RoofMatl'])

Exterior2nd = LabelEncoder()
dataset['Exterior2nd'] = Exterior2nd.fit_transform(dataset['Exterior2nd'])

MasVnrType = LabelEncoder()
dataset['MasVnrType'] = MasVnrType.fit_transform(dataset['MasVnrType'])

ExterQual = LabelEncoder()
dataset['ExterQual'] = ExterQual.fit_transform(dataset['ExterQual'])

Foundation = LabelEncoder()
dataset['Foundation'] = Foundation.fit_transform(dataset['Foundation'])

BsmtQual = LabelEncoder()
dataset['BsmtQual'] = BsmtQual.fit_transform(dataset['BsmtQual'])

BsmtCond = LabelEncoder()
dataset['BsmtCond'] = BsmtCond.fit_transform(dataset['BsmtCond'])

BsmtExposure = LabelEncoder()
dataset['BsmtExposure'] = BsmtExposure.fit_transform(dataset['BsmtExposure'])

BsmtFinType1 = LabelEncoder()
dataset['BsmtFinType1'] = BsmtFinType1.fit_transform(dataset['BsmtFinType1'])

BsmtFinType2 = LabelEncoder()
dataset['BsmtFinType2'] = BsmtFinType2.fit_transform(dataset['BsmtFinType2'])

KitchenQual = LabelEncoder()
dataset['KitchenQual'] = KitchenQual.fit_transform(dataset['KitchenQual'])

Functional = LabelEncoder()
dataset['Functional'] = Functional.fit_transform(dataset['Functional'])

FireplaceQu = LabelEncoder()
dataset['FireplaceQu'] = FireplaceQu.fit_transform(dataset['FireplaceQu'])

GarageFinish = LabelEncoder()
dataset['GarageFinish'] = GarageFinish.fit_transform(dataset['GarageFinish'])

GarageQual = LabelEncoder()
dataset['GarageQual'] = GarageQual.fit_transform(dataset['GarageQual'])

GarageCond = LabelEncoder()
dataset['GarageCond'] = GarageCond.fit_transform(dataset['GarageCond'])

SaleCondition = LabelEncoder()
dataset['SaleCondition'] = SaleCondition.fit_transform(dataset['SaleCondition'])



# create dummy variables for categorical data and exclude 1 one of them to avoid the dummy variable trap
dataset = pd.get_dummies(dataset, columns=['MSZoning', 'Street', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofMatl', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'SaleCondition'], drop_first=True)


# Splitting the dataset into Train and Test
dataset_train = dataset.iloc[:1460, :]
dataset_test = dataset.iloc[1460:, :]

# Splitting the dataset into the input and output
X_train = dataset_train.iloc[:,dataset_train.columns != 'Id' ]
X_train = X_train.drop('SalePrice', axis = 1)
y_train = dataset_train.iloc[: , dataset_train.columns == 'SalePrice']
X_test = dataset_test.iloc[:,dataset_train.columns != 'Id' ]
X_test = X_test.drop('SalePrice', axis = 1)

#getting the PassIndex for the Submission dataset
pass_index = dataset.iloc[1460:,0]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)





















# Part 2 - Making the predictions
from sklearn.ensemble import GradientBoostingRegressor
reg = GradientBoostingRegressor()
reg.fit(X_train,y_train)

# Predicting the Test set results
y_pred = reg.predict(X_test)


#Part 3 - Generating Submission File
only_final_values = pd.DataFrame(columns = ['Id', 'SalePrice'])
only_final_values['Id'] = pass_index
only_final_values['SalePrice'] = y_pred
only_final_values['SalePrice'] = only_final_values['SalePrice'].astype(np.float64)
only_final_values.to_csv('house_prices_submission_final.csv', index = False)



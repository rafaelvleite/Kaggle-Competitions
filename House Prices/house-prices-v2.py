# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Part 1 - Preprocessing the data

# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import stats
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder


# Importing the test dataset
dataset = pd.read_csv('train+test_cleaned.csv')
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')


# Separate quantitative and qualitative features and perform suitable tests on both to find out important features.

quan = [c for c in train.columns if train[c].dtype != 'object']    #quantitative variables
qual = [c for c in train.columns if train[c].dtype == 'object']    #qualitative variables

#for quantitative variables we will perform pearson correlation test

corr_mat = train[quan].corr()
plt.figure(figsize = (12,9))
sns.heatmap(corr_mat)


#let's select top 20 features based on correlation with SalePrice and observe them.
top_20_quant = corr_mat.nlargest(20,'SalePrice')['SalePrice'].index
corr_coff_q = train[top_20_quant].corr()       #correlation cofficients of top 20 quantitative features
plt.figure(figsize = (12,9))
sns.heatmap(corr_coff_q , annot = True)


#selected quantitative features

quan_s = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt','YearRemodAdd', 'MasVnrArea', 'Fireplaces',
       'BsmtFinSF1', 'LotFrontage', 'WoodDeckSF','OpenPorchSF', 'HalfBath']


# to analyse qualitative features , we will first fill the missing values
def fill_missing(data , features):
    for col in features:
        data[col] = data[col].astype('category')
        data[col] = data[col].cat.add_categories('MISSING')
        data[col] = data[col].fillna('MISSING')
    
df = pd.concat([train.drop('SalePrice',axis=1),test])
print ("missing values in qualitative features before "+ str(df[qual].isnull().sum().sum()))
fill_missing(df , qual)
print ("missing values in qualitative features after "+ str(df[qual].isnull().sum().sum()))

y = train['SalePrice']
train = df[0:1460]    #updating train and test after filing missing values
test = df[1460:]

# Now to select most important features we wil perform annova test on qualitative features

def annov(df , features):
    ann = pd.DataFrame()
    ann['feat'] = features
    pvals = []
    for col in features:
        var = []
        for val in df[col].unique():
            var.append(df[df[col] == val].SalePrice.values)
        pvals.append(stats.f_oneway(*var).pvalue)
    ann['pvals'] = pvals
    return ann

train['SalePrice'] = y
result = annov(train , qual)


# We will use pvalues less than .05 only and convert those pvalues to disparities for better visualization. More disparity means less pcalues which means more significance of a feature.
# we will again select top 20 qualitative features

result['disp'] = np.log(1. / result['pvals'])
result = result.sort_values('disp' ,ascending = False)
plt.figure(figsize = (12,9))
sns.barplot(y = result['feat'] , x = result['disp'] ,orient = 'h')


qual_s = result.feat[0:20]      #already sorted

#we need to encode the qualitative features and convert to int to perform statistics tests on them 
df = pd.concat([train,test])

def encode(data , features):
    encoder=LabelEncoder()
    qual_E = []
    for col in features:
        encoder.fit(data[col].unique())
        data[col+'_E'] = encoder.transform(data[col])
        qual_E.append(col+'_E')
    return qual_E
    
qual_E_s = encode(df,qual_s)

df_s = df[quan_s + qual_E_s]    
train = df_s[0:1460]
test = df_s[1460:]


#chi square test between categorical features

def chi_sq(data , features):
    chi = pd.DataFrame(index = features , columns = features)
    for col1 in features:
        for col2 in features:
            #reshape beacause chi2 accepts dataframe of shape(n_samples,n_fetures)
            chi.loc[col1,col2] = chi2(data[col1].values.reshape(-1,1),data[col2])[1]
    return chi


chi = chi_sq(train , qual_E_s)
plt.figure(figsize = (12,9))
chi = chi.astype(float)
sns.barplot( y = chi.index , x= chi['Neighborhood_E'])


qual_E_final = ['Neighborhood_E' , 'SaleCondition_E' , 'SaleType_E' , 'GarageCond_E']
X = train[quan_s + qual_E_final]
X_test = test[quan_s + qual_E_final]

#filling missing values
data = pd.concat([X,X_test])
data.isnull().sum().sort_values(ascending = False)[0:10]


# Width of street will be same for neighbors , so we can impute lotfrontage from lotfrontage of neighbors

missing_lot = data.groupby('Neighborhood_E').LotFrontage.mean()

data.index = (range(0,data.shape[0]))

for index in range(0,data.shape[0]):
    if np.isnan(data.loc[index,'LotFrontage']):
        data.loc[index,'LotFrontage'] = missing_lot[data.loc[index,'Neighborhood_E']]


data.GarageCars =data.GarageCars.fillna(2.)          #mode
data.TotalBsmtSF =data.TotalBsmtSF.fillna(data.TotalBsmtSF.mean())
data.BsmtFinSF1 =data.BsmtFinSF1.fillna(data.BsmtFinSF1.mean())

data.MasVnrArea.describe()

#null value in MasVnrArea indicates absence of MasVnr , so we will fill it with zero

data['MasVnrArea'] = data['MasVnrArea'].fillna(0.)

print('Total missing values in dataset = ' + str(data.isnull().sum().sum()))


X_train = data[0:1460]
X_test = data[1460:]



# Part 2 - Making the predictions and Generating Submission File
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0, max_depth = 8, min_samples_split = 5, min_samples_leaf = 5)
classifier.fit(X_train, y.values.ravel())

# Predicting the Test set results
y_pred = classifier.predict(X_test)

sample = pd.read_csv('sample_submission.csv')
sample['SalePrice'] = y_pred
sample.to_csv('sol.csv',index=False)












reg = GradientBoostingRegressor()
reg.fit(X_train,y)
sample = pd.read_csv('sample_submission.csv')
sample['SalePrice'] = reg.predict(X_test)
sample.to_csv('sol.csv',index=False)





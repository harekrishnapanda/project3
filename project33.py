# -*- coding: utf-8 -*-
"""
Created on Wed Sep 9 10:22:05 2018

@author: Harekrishna
"""
# Importing the liabraries
import numpy as np
import pandas as pd 

# Reading the data
df = pd.read_csv("adult.data",na_values="?", skipinitialspace=True)
#cheking the inpu data for null values
df.info()   
df.isnull().sum()

# Renaming the columns correctly
df.columns = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education-num', 'Marital-status', \
              'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss', \
              'Hours-per-week', 'Native-country', 'Earning']


# Checking the NA values
df.isnull().sum()

# Dropping the NA values in below 3 categorical columns as its only 7% of the data
# the ‘workclass’ ‘occupation’ ‘nativecountry’ 

df_new = df.dropna()

# Checking the NA values
df_new.isnull().sum()

# applying lebel encoding on these columns.
# Fist print the unique values of each features.
cat_cols = ['Workclass', 'Education', 'Marital-status', 'Occupation', 'Relationship',\
                'Race', 'Sex', 'Native-country', 'Earning' ]
'''
print("Numbers of unique values in each catagorical columns are: ")

for feature in cat_cols:
    print(feature + " has total : " + str(len(df_new[feature].unique())))
'''
# Applying label encoding on only catagorical columns, i.e. preferred_foot
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

for cols in cat_cols:
    df_new[cols] = label_encoder.fit_transform(df_new[cols])
    

# Data after one hot encodng of all catagorical data
df_new.head()

# splitting the data set into X and Y Axis
X = df_new.iloc[:,:-1]
y = df_new.Earning

# Splitting the data into train and test data by 85:15 ratio
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=0)


# building the optimal model using backward elimination
# SL = 0.05 and eliminating those features which have p > SL
import statsmodels.formula.api as sm
X_train = np.append(arr = np.ones((25636,1)).astype(int), values = X_train, axis = 1)
X_train_opt = X_train[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_train_opt).fit()
regressor_OLS.summary()

# Here We can see Native-country is having P value > 0.05, so can be eliminated now.
X_train.drop(['Native-country'], axis=1, inplace=True)
X_test.drop(['Native-country'], axis=1, inplace=True)

# Checking the P values again if any other column can be eliminated
X_train_opt = X_train[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_train_opt).fit()
regressor_OLS.summary()
# No more eleminations are required based on the P value

# Logistic Regression 
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
scor=classifier.score(X_train,y_train)
scor

# Cross validation
from sklearn.cross_validation import cross_val_score
cv = cross_val_score(estimator = classifier, X=X,y=y,scoring='accuracy',cv=50)
print(cv.mean())

# DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
dtree.fit(X_train, y_train)
tscore=dtree.score(X_test, y_test)
print(tscore)

# RandomForestRegressor

# Try different numbers of n_estimators - this will take a while or so
from sklearn.ensemble import RandomForestClassifier
regr_rf = RandomForestClassifier(max_depth=30, random_state=2)

estimators = np.arange(100, 200, 10)
scores = []
for n in estimators:
    regr_rf.set_params(n_estimators=n)
    regr_rf.fit(X_train, y_train)
    scores.append(regr_rf.score(X_test, y_test))
    #print(scores)
#max_sc_idx = scores.index(max(scores))
print(max(scores))



# Bagging
from sklearn.ensemble import BaggingClassifier
bg=BaggingClassifier(DecisionTreeClassifier(),n_estimators=20, max_samples = 0.5, max_features=1.0)
bg.fit(X_train,y_train)
bgscore=bg.score(X_test,y_test)
print(bgscore)

# Boosting
from sklearn.ensemble import AdaBoostClassifier
bo=AdaBoostClassifier(n_estimators=50, learning_rate=1.)
bo.fit(X_train,y_train)
boscore=bo.score(X_test,y_test)
print(bgscore)

#fitting XGBoost to the Training set
import xgboost
classifier = xgboost.XGBClassifier()
classifier.fit(X_train,y_train)
cscore=classifier.score(X_train,y_train)
print(cscore)

##########################  Naive Bayes  ####################################################################

# Fitting Naive Bayes classifier to the opt Training set
from sklearn.naive_bayes import GaussianNB
nbclassifier = GaussianNB()
nbclassifier.fit(X_train, y_train)
nbscore=nbclassifier.score(X_train,y_train)
print(nbscore)


############################  Support Vector Machine (SVM)  ################################################  

# Fitting Support Vector Machine (SVM) to the opt Training set
from sklearn.svm import SVC
svmclassifier = SVC(kernel = 'rbf')
svmclassifier.fit(X_train, y_train)
svmscore=svmclassifier.score(X_train,y_train)
print(svmscore)

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 9 10:22:05 2018
@author: Harekrishna
"""
# Importing the liabraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading the data
df = pd.read_csv("adult.data",na_values="?", skipinitialspace=True)
#cheking the inpu data for null values
df.info()   

# Renaming the columns correctly
df.columns = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education-num', 'Marital-status', \
              'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss', \
              'Hours-per-week', 'Native-country', 'Earning']

# Data Visualization
fig = plt.figure(figsize=(18,10))

plt.subplot2grid((2,3),(0,0))
df.Workclass.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Workclass")

plt.subplot2grid((2,3),(0,1))
df.Earning.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Earning")

plt.subplot2grid((2,3),(0,2))
df.Occupation.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Occupation")

plt.show()

# Checking the NA values
df.isnull().sum()

# Dropping the NA values in below 3 categorical columns as its only 7% of the data
# the ‘workclass’ ‘occupation’ ‘nativecountry’ 
df_new = df.dropna()
##########df_new = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
##########df.dropna(subset=['Workclass','Occupation','Native-country'],how='all').shape

# Checking the NA values
df_new.isnull().sum()

# applying lebel encoding on these columns.
cat_cols = ['Workclass', 'Education', 'Marital-status', 'Occupation', 'Relationship',\
                'Race', 'Sex', 'Native-country', 'Earning' ]

# Applying label encoding on only catagorical columns, i.e. preferred_foot
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

for cols in cat_cols:
    df_new[cols] = label_encoder.fit_transform(df_new[cols])

# Getting the corelation matrix to see which columns effect the Earning    
df_corr= df_new.corr()
print('From the co_relation matrix it is found that following columns have greater impact on Earnings : Age, Education_num, Sex, Capital_gain, ours_per_week')

# splitting the data set into X and Y Axis
X = df_new.iloc[:,:-1]
y = df_new.Earning

# Splitting the data into train and test data by 85:15 ratio
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# building the optimal model using backward elimination
# SL = 0.05 and eliminating those features which have p > SL
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((30161,1)).astype(int), values = X, axis = 1)
X_opt = X[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# Checking the P values again if any other column can be eliminated
# No more eleminations are required based on the P value
X_opt = X[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


# Here We can see Native-country is having P value > 0.05, so can be eliminated now.
X_train.drop(['Native-country'], axis=1, inplace=True)
X_test.drop(['Native-country'], axis=1, inplace=True)



##########################  Naive Bayes  ####################################################################

# Fitting Naive Bayes classifier to the opt Training set
from sklearn.naive_bayes import GaussianNB
nbclassifier = GaussianNB()
nbclassifier.fit(X_train, y_train)
nbscore=nbclassifier.score(X_train,y_train)
print('score with Naive Bayes is ',+ nbscore)

'''
############################  Support Vector Machine (SVM)  ################################################  
# Fitting Support Vector Machine (SVM) to the opt Training set
from sklearn.svm import SVC
svmclassifier = SVC(kernel = 'rbf')
svmclassifier.fit(X_train, y_train)
svmscore=svmclassifier.score(X_train,y_train)
print(svmscore)
'''
############################  Logistic Regression  ################################################
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
lrscore=classifier.score(X_test,y_test)
print('score with LogisticRegression is ',+ lrscore)

############################  cross validation  ################################################
from sklearn.cross_validation import cross_val_score
cv = cross_val_score(estimator = classifier, X=X,y=y,scoring='accuracy',cv=50)
print('score with cross validation is ',+ cv.mean())

############################  DecisionTreeClassifier  ################################################
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
dtree.fit(X_train, y_train)
tscore=dtree.score(X_test, y_test)
print('score with DecisionTreeClassifier is ',+ tscore)

############################  RandomForestRegressor  ################################################
# Trying here to get best value of n_estimators
from sklearn.ensemble import RandomForestClassifier
rfclassifier = RandomForestClassifier(max_depth=30, random_state=2)

estimators = np.arange(100, 300, 10)
rfscore = []
for n in estimators:
    rfclassifier.set_params(n_estimators=n)
    rfclassifier.fit(X_train, y_train)
    rfscore.append(rfclassifier.score(X_test, y_test))
print('score with RandomForestRegressor is ',+ max(rfscore))

############################  xgboost  ################################################
import xgboost
xgclassifier = xgboost.XGBClassifier()
xgclassifier.fit(X_train,y_train)
xgclassifier.predict(X_test)
xgscore=xgclassifier.score(X_test,y_test)
print('score with XGBoost is ',+ xgscore)

print('The best algorith suited for this dataset is the XGBOOST Classifier')

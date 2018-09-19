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
df.info()   # df.isnull().sum()

# Renaming the columns correctly
df.columns = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education-num', 'Marital-status', \
              'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss', \
              'Hours-per-week', 'Native-country', 'Earning']

rows_before_droppinig = df.shape[0]
# Printing all the columns
print(df.columns)

# Checking the NA values
df.isnull().sum()
df1= df.corr()

# Dropping the NA values which is actually 7.368% of total Data
# It is observed that in most of the missing data set, 
# the ‘workclass’ variable and ‘occupation’ variable are missing data together. 
# And the remaining have ‘nativecountry’ variable missing. We could handle the 
# missing values by imputing the data. However, since ‘workclass’, ‘occupation’ 
# and ‘nativecountry’ could potentially be very good predictors of income, 
# imputing may simply skew the model.

df_opt = df.dropna()
rows_after_droppinig = df_opt.shape[0]

# Checking the NA values
df_opt.isnull().sum()

# number of rows dropped
diff = rows_before_droppinig - rows_after_droppinig
print("Total number of rows dropped is {} wich is {:.4}% of total rows."   \
              .format(diff,(diff/rows_before_droppinig)*100))
print('-'*50)

# applying one hot and lebel encoding on these columns.
# Fist print the unique values of each features.
cat_features = ['Workclass', 'Education', 'Marital-status', 'Occupation', 'Relationship',\
                'Race', 'Sex', 'Native-country', 'Earning' ]
print("Total numbers of unique values in catagorical features are: ")

for feature in cat_features:
    print(feature + "has total : " + str(len(df[feature].unique())))

# Applying One hot encoding on only catagorical columns, i.e. preferred_foot
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

for feature in cat_features:
    #feature = 'Workclass'
    df_opt[feature] = label_encoder.fit_transform(df_opt[feature])
    

# Data after one hot encodng of all catagorical data
df_opt.head()

# Defining X and y from the Dataset
X = df_opt.iloc[:,:-1]
y = df_opt.Earning

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.10,random_state=0)

backwardEle(X, y)

# Here We can see X14 i.e. "Native-country" is having P value > 0.05
# o better to eleminate this column
X_train.drop(['Native-country'], axis=1, inplace=True)

# Same time drop this column from test set as well
X_test.drop(['Native-country'], axis=1, inplace=True)

X_train.head()

backwardEle(X, y)

# After eleminating 'Native-country', there is no more columns which are having P value > 0.05
# i.e. No more eleminations are required.

# Applying Logistic regression
# Logistic Regression 

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
scor=classifier.score(X_train,y_train)
scor


# improvement required
# Cross validation from SKlearn

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

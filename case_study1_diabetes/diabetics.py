""" 
Summary of the program:
1. import lib's
2. read dataset
3. distribution of all the features/variables
4. skewness and outliers in data distribution is observed
5. address skewness and outliers in data distribution
6. check for multi-collinearity
7. divide dataset into train & test
8. fit logistic regression model
9. test model on test dataset
10. compute accuracy score
11. increase accuracy using hyperparameter tuning with grid search 
12. Use k-fold CV for generalizing on dataset
13. Dump ML model and StandardScaler to local file
 """

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("case_study1_diabetes/diabetes.csv")

data.head()
""" 
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome
0            6      148             72             35        0  33.6                     0.627   50        1
1            1       85             66             29        0  26.6                     0.351   31        0
2            8      183             64              0        0  23.3                     0.672   32        1
3            1       89             66             23       94  28.1                     0.167   21        0
4            0      137             40             35      168  43.1                     2.288   33        1 """

data.describe()
""" 
       Pregnancies     Glucose  BloodPressure  SkinThickness     Insulin         BMI  DiabetesPedigreeFunction         Age     Outcome
count   768.000000  768.000000     768.000000     768.000000  768.000000  768.000000                768.000000  768.000000  768.000000
mean      3.845052  120.894531      69.105469      20.536458   79.799479   31.992578                  0.471876   33.240885    0.348958
std       3.369578   31.972618      19.355807      15.952218  115.244002    7.884160                  0.331329   11.760232    0.476951
min       0.000000    0.000000       0.000000       0.000000    0.000000    0.000000                  0.078000   21.000000    0.000000
25%       1.000000   99.000000      62.000000       0.000000    0.000000   27.300000                  0.243750   24.000000    0.000000
50%       3.000000  117.000000      72.000000      23.000000   30.500000   32.000000                  0.372500   29.000000    0.000000
75%       6.000000  140.250000      80.000000      32.000000  127.250000   36.600000                  0.626250   41.000000    1.000000
max      17.000000  199.000000     122.000000      99.000000  846.000000   67.100000                  2.420000   81.000000    1.000000

It seems that there are no missing values in our data. Great, let's see the distribution of data:
 """

# let's see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in data:
    if plotnumber<=9 :     # as there are 9 columns in the data
        ax = plt.subplot(3,3,plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column,fontsize=20)
        #plt.ylabel('Salary',fontsize=20)
    plotnumber+=1
plt.show()

""" 
- We can see there is some skewness in the data, let's deal with data.
- Also, we can see there few data for columns Glucose, Insulin, skin thickness, BMI and Blood Pressure which have
 value as 0. That's not possible. You can do a quick search to see that one cannot have 0 values for these.
  Let's deal with that. we can either remove such data or simply replace it with their respective mean values. 
  Let's do the latter.
 """

# replacing zero values with the mean of the column
data['BMI'] = data['BMI'].replace(0,data['BMI'].mean())
data['BloodPressure'] = data['BloodPressure'].replace(0,data['BloodPressure'].mean())
data['Glucose'] = data['Glucose'].replace(0,data['Glucose'].mean())
data['Insulin'] = data['Insulin'].replace(0,data['Insulin'].mean())
data['SkinThickness'] = data['SkinThickness'].replace(0,data['SkinThickness'].mean())

# let's see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in data:
    if plotnumber<=9 :
        ax = plt.subplot(3,3,plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column,fontsize=20)
        #plt.ylabel('Salary',fontsize=20)
    plotnumber+=1
plt.show()

# box plot for outliers view
fig, ax = plt.subplots(figsize=(15,10))
sns.boxplot(data=data, width= 0.5,ax=ax,  fliersize=3)
plt.show()

# fix outliers
q = data['Pregnancies'].quantile(0.98)

# we are removing the top 2% data from the Pregnancies column
data_cleaned = data[data['Pregnancies']<q]
q = data_cleaned['BMI'].quantile(0.99)

# we are removing the top 1% data from the BMI column
data_cleaned  = data_cleaned[data_cleaned['BMI']<q]
q = data_cleaned['SkinThickness'].quantile(0.99)

# we are removing the top 1% data from the SkinThickness column
data_cleaned  = data_cleaned[data_cleaned['SkinThickness']<q]
q = data_cleaned['Insulin'].quantile(0.95)

# we are removing the top 5% data from the Insulin column
data_cleaned  = data_cleaned[data_cleaned['Insulin']<q]
q = data_cleaned['DiabetesPedigreeFunction'].quantile(0.99)

# we are removing the top 1% data from the DiabetesPedigreeFunction column
data_cleaned  = data_cleaned[data_cleaned['DiabetesPedigreeFunction']<q]
q = data_cleaned['Age'].quantile(0.99)

# we are removing the top 1% data from the Age column
data_cleaned  = data_cleaned[data_cleaned['Age']<q]

# let's see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in data_cleaned:
    if plotnumber<=9 :
        ax = plt.subplot(3,3,plotnumber)
        sns.distplot(data_cleaned[column])
        plt.xlabel(column,fontsize=20)
        #plt.ylabel('Salary',fontsize=20)
    plotnumber+=1
plt.show()

X = data.drop(columns = ['Outcome'])
y = data['Outcome']

# let's see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in X:
    if plotnumber<=9 :
        ax = plt.subplot(3,3,plotnumber)
        sns.stripplot(y,X[column])
    plotnumber+=1
plt.tight_layout()
plt.show()

# Great!! Let's proceed by checking multicollinearity in the dependent variables. Before that, 
# we should scale our data. Let's use the standard scaler for that.

scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(X_scaled,i) for i in range(X_scaled.shape[1])]
vif["Features"] = X.columns

#let's check the values
vif
""" vif	Features
0	1.431075	Pregnancies
1	1.347308	Glucose
2	1.247914	BloodPressure
3	1.450510	SkinThickness
4	1.262111	Insulin
5	1.550227	BMI
6	1.058104	DiabetesPedigreeFunction
7	1.605441	Age """

x_train,x_test,y_train,y_test = train_test_split(X_scaled,y, test_size= 0.25)

# let's fit the data into kNN model and see how well it performs:
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
""" 
KNeighborsClassifier() """

y_pred = knn.predict(x_test)

knn.score(x_train,y_train)
# 0.8368055555555556

print("The accuracy score is : ", accuracy_score(y_test,y_pred))
# The accuracy score is :  0.765625

# Let's try to increase the accuracy by using hyperparameter tuning.

param_grid = { 'algorithm' : ['ball_tree', 'kd_tree', 'brute'],
               'leaf_size' : [18,20,25,27,30,32,34],
               'n_neighbors' : [3,5,7,9,10,11,12,13]
              }

gridsearch = GridSearchCV(knn, param_grid,verbose=3)

gridsearch.fit(x_train,y_train)
""" 
GridSearchCV(estimator=KNeighborsClassifier(),
             param_grid={'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                         'leaf_size': [18, 20, 25, 27, 30, 32, 34],
                         'n_neighbors': [3, 5, 7, 9, 10, 11, 12, 13]},
             verbose=3) """

# let's see the  best parameters according to gridsearch
gridsearch.best_params_
# {'algorithm': 'ball_tree', 'leaf_size': 18, 'n_neighbors': 11}

# we will use the best parameters in our k-NN algorithm and check if accuracy is increasing.
knn = KNeighborsClassifier(algorithm = 'ball_tree', leaf_size =18, n_neighbors =11)
knn.fit(x_train,y_train)
# KNeighborsClassifier(algorithm='ball_tree', leaf_size=18, n_neighbors=11)

knn.score(x_train,y_train)
# 0.8072916666666666

""" 
- Looks like accuracy for training has decreased, maybe our model was overfitting the data before. 
- Let's see how it performs on the test data. """

knn.score(x_test,y_test)
# 0.734375

# Great, accuracy score has increased for our test data. So, indeed our model was overfitting before. 

# Let's now use k-fold cross validation and check how well our model is generalizing over our dataset:
#    We are randomly selecting our k to be 12 for k fold.

#k-fold cross validation 
kfold = KFold(n_splits=12,random_state= 42,shuffle=True)
kfold.get_n_splits(X_scaled)
# 12

from statistics import mean
knn = KNeighborsClassifier(algorithm = 'ball_tree', leaf_size =18, n_neighbors =11)
cnt =0
count=[]
train_score =[]
test_score = []

for train_index,test_index in kfold.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index] # our scaled data is an array so it can work on x[value]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index] # y is a dataframe so we have to use "iloc" to retreive data
    knn.fit(X_train,y_train)
    train_score_ = knn.score(X_train,y_train)
    test_score_ =  knn.score(X_test,y_test)
    cnt+=1
    count.append(cnt)
    train_score.append(train_score_)
    test_score.append(test_score_)
    
    print("for k = ", cnt)
    print("train_score is :  ", train_score_, "and test score is :  ", test_score_)
""" 
KNeighborsClassifier(algorithm='ball_tree', leaf_size=18, n_neighbors=11)
for k =  1
train_score is :   0.7982954545454546 and test score is :   0.6875
KNeighborsClassifier(algorithm='ball_tree', leaf_size=18, n_neighbors=11)
for k =  2
train_score is :   0.7855113636363636 and test score is :   0.84375
KNeighborsClassifier(algorithm='ball_tree', leaf_size=18, n_neighbors=11)
for k =  3
train_score is :   0.8011363636363636 and test score is :   0.703125
KNeighborsClassifier(algorithm='ball_tree', leaf_size=18, n_neighbors=11)
for k =  4
train_score is :   0.8039772727272727 and test score is :   0.75
KNeighborsClassifier(algorithm='ball_tree', leaf_size=18, n_neighbors=11)
for k =  5
train_score is :   0.7883522727272727 and test score is :   0.84375
KNeighborsClassifier(algorithm='ball_tree', leaf_size=18, n_neighbors=11)
for k =  6
train_score is :   0.8068181818181818 and test score is :   0.734375
KNeighborsClassifier(algorithm='ball_tree', leaf_size=18, n_neighbors=11)
for k =  7
train_score is :   0.7911931818181818 and test score is :   0.765625
KNeighborsClassifier(algorithm='ball_tree', leaf_size=18, n_neighbors=11)
for k =  8
train_score is :   0.8068181818181818 and test score is :   0.65625
KNeighborsClassifier(algorithm='ball_tree', leaf_size=18, n_neighbors=11)
for k =  9
train_score is :   0.7940340909090909 and test score is :   0.828125
KNeighborsClassifier(algorithm='ball_tree', leaf_size=18, n_neighbors=11)
for k =  10
train_score is :   0.8025568181818182 and test score is :   0.75
KNeighborsClassifier(algorithm='ball_tree', leaf_size=18, n_neighbors=11)
for k =  11
train_score is :   0.7897727272727273 and test score is :   0.75
KNeighborsClassifier(algorithm='ball_tree', leaf_size=18, n_neighbors=11)
for k =  12
train_score is :   0.796875 and test score is :   0.71875 """

print("************************************************")
print("************************************************")
print("Average train score is :  ", mean(train_score))
print("Average test score is :  ", mean(test_score))

""" ************************************************
************************************************
Average train score is :   0.7971117424242424
Average test score is :   0.7526041666666666 """

# let's plot the test_accuracy with the value of k in k-fold
plt.plot(count,test_score)
plt.xlabel('Value of K for k-fold')
plt.ylabel('test accuracy')
plt.xticks(np.arange(0, 12, 1)) 
plt.yticks(np.arange(0.65, 1, 0.05)) 
plt.show()

""" 
- Our cross validation tells that on an average our model has a 75% accuracy on our test data. 
so, that's how we can use cross validation to compute how well our model is generalizing on our data.

We can also use cross valdition score to opt between different models or to do hyperparameter tuning.
 """

# let's save the model
import pickle

with open('case_study1_diabetes/modelForPrediction.sav', 'wb') as f:
    pickle.dump(knn,f)
    
with open('case_study1_diabetes/standardScalar.sav', 'wb') as f:
    pickle.dump(scalar,f)
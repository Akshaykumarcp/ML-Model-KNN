# load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# data preprocessing

# define column names
names = ['x', 'y', 'class']

# load training data
df = pd.read_csv('dataset/3.concertriccir2.csv', header=None, names=names)
print(df.head())

# create design matrix X and target vector y
X = np.array(df.iloc[:, 0:2]) # end index is exclusive
y = np.array(df['class']) # showing you two ways of indexing a pandas df

# split the data set into train and test
X_1, X_test, y_1, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# split the train data set into cross validation train and cross validation test
X_tr, X_cv, y_tr, y_cv = train_test_split(X_1, y_1, test_size=0.3)

# SIMPLE CROSS VALIDATION

for i in range(1,30,2):
    # instantiate learning model (k = 30)
    knn = KNeighborsClassifier(n_neighbors=i)

    # fitting the model on crossvalidation train
    knn.fit(X_tr, y_tr)

    # predict the response on the crossvalidation train
    pred = knn.predict(X_cv)

    # evaluate CV accuracy
    acc = accuracy_score(y_cv, pred, normalize=True) * float(100)
    print('\nCV accuracy for k = %d is %d%%' % (i, acc))

""" CV accuracy for k = 1 is 85%
KNeighborsClassifier(n_neighbors=3)

CV accuracy for k = 3 is 84%
KNeighborsClassifier()

CV accuracy for k = 5 is 80%
KNeighborsClassifier(n_neighbors=7)

CV accuracy for k = 7 is 77%
KNeighborsClassifier(n_neighbors=9)

CV accuracy for k = 9 is 78%
KNeighborsClassifier(n_neighbors=11)

CV accuracy for k = 11 is 77%
KNeighborsClassifier(n_neighbors=13)

CV accuracy for k = 13 is 74%
KNeighborsClassifier(n_neighbors=15)

CV accuracy for k = 15 is 69%
KNeighborsClassifier(n_neighbors=17)

CV accuracy for k = 17 is 68%
KNeighborsClassifier(n_neighbors=19)

CV accuracy for k = 19 is 69%
KNeighborsClassifier(n_neighbors=21)

CV accuracy for k = 21 is 66%
KNeighborsClassifier(n_neighbors=23)

CV accuracy for k = 23 is 62%
KNeighborsClassifier(n_neighbors=25)

CV accuracy for k = 25 is 61%
KNeighborsClassifier(n_neighbors=27)

CV accuracy for k = 27 is 60%
KNeighborsClassifier(n_neighbors=29)

CV accuracy for k = 29 is 60% """

knn = KNeighborsClassifier(1)
knn.fit(X_tr,y_tr)
pred = knn.predict(X_test)
acc = accuracy_score(y_test, pred, normalize=True) * float(100)
print('Test accuracy for k = 1 is %d%%' % (acc))
# Test accuracy for k = 1 is 88%

# 10 FOLD CROSS VALIDATION

# creating odd list of K for KNN
myList = list(range(0,50))
neighbors = list(filter(lambda x: x % 2 != 0, myList))

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_tr, y_tr, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print('\nThe optimal number of neighbors is %d.' % optimal_k)

# plot misclassification error vs k 
plt.plot(neighbors, MSE)

for xy in zip(neighbors, np.round(MSE,3)):
    plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.savefig("0.10_Misclassification_Error.png")
plt.show()

print("the misclassification error for each k value is : ", np.round(MSE,3))

#  KNN with k = optimal_k
# instantiate learning model k = optimal_k
knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)

# fitting the model
knn_optimal.fit(X_tr, y_tr)

# predict the response
pred = knn_optimal.predict(X_test)

# evaluate accuracy
acc = accuracy_score(y_test, pred) * 100
print('\nThe accuracy of the knn classifier for k = %d is %f%%' % (optimal_k, acc))
# The accuracy of the knn classifier for k = 1 is 88.666667%


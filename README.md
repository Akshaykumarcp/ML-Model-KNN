## Content

1. About KNN Model
2. KNN Mind Map image
3. KNN working
4. Distance measures
5. Limitations of kNN
6. Failure of KNN
7. Decision surface for KNN
8. KNN when Overfit & underfit
9. Determine right K value in KNN
10. Can KNN do Multiclass classification ?
11. Can KNN do Probabilitics predictions ?
12. KNN when dataset is imbalanced
13. KNN when dataset has outliers 
14. Is KNN affected by scale of the features ?
15. Are kNN interpetable ?
16. KNN techniques to reduce space & time complexity
17. Lazy Learner
18. Weighted Nearest Neighbor
19. Pros of KNN
20. Acknowledgements
21. License

# 1. About KNN Model
- Simple & powerfull
- For classification & regression
- Distance based machine learning algorithm

# 2. KNN Mind Mapping
![KNN mind mapping](https://github.com/Akshaykumarcp/ML-Model-KNN/blob/main/KNN.jpg)

# 3. KNN Working

![KNN Working](http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1531424125/KNN_final1_ibdm8a.png)

- Select the value for k (ex: 1,3,5,7 etc)
- Compute distance between point to be classified & every other point in training dataset
- Choose k nearest data points
- Run majority vote among selected data points, dominating classification is the winner and winner is the classified point
- Repeat...

# 4. Distance Measures:
    
## 4.1. Euclidean Distance
    - is the shortest length between two points
    - often written as || x1 - x2 || --> length of x1 - x2
    - is the L2 norm

![Euclidean distance](https://cdn-images-1.medium.com/max/800/1*ZrwEraj9S-u_KOWdKWc8sQ.png)

## 4.2. Manhattan Distance
    - is the absolute value of length
    - Often written as | x1 - x2 | - absolute value of x1 - x2
    - is the L1 norm

![Manhattan distance](https://d1m75rqqgidzqn.cloudfront.net/wp-data/2020/07/01130416/june-30-hierarchical-clustering-infograph-for-blog-4.png)

In summary:

- L1 norm is absolute value
- L2 norm is subtracting and squaring

L1 & L2 norm are computed for vectors.

## 4.3. Minkowski Distance
    - is the generalization of L1 & L2 norm
    - Depending on p value, distance is categorized

Formula:
![Minkowski Distance](https://rittikghosh.com/images/min.png)

[Source](https://rittikghosh.com/images/min.png)

When p = 1, known as Manhattan Distance

When p = 2, known as Euclidean Distance


![Minkowski Distance](https://miro.medium.com/max/1400/1*zJ8PPj8mO4gO2U9-MlwfrA.jpeg)
[Source](https://miro.medium.com/max/1400/1*zJ8PPj8mO4gO2U9-MlwfrA.jpeg)

Note: Distances are always for 2 points & norms are always for  vectors

## 4.4. Hamming Distance
    - is the distance between boolean/binary vector
    - used in text processing (binary BOW) and having boolean vectors

Example:
![Hamming Distance](https://www.researchgate.net/profile/Fredrick-Ishengoma/publication/264978395/figure/fig1/AS:295895569584128@1447558409105/Example-of-Hamming-Distance.png)

# 5. Limitations of KNN

- Time (since the model is not saved beforehand in this algorithm (lazy learner), so every time one predicts a test value, it follows the same steps again and again) & space (store the whole training set for every test set) complexity is high
- Hence not widely used
- Not suitable for high dimensional data.
- Expensive in testing phase

# 6. Failure of KNN
- When query point is far away from the k nearest neighbours at the time of training and prediction, target feature (yi) is not sure of training/prediction.
- Data points based on target features are randomly spread i,e in binary classification posivte and negative data points are jumbled hence no usefull information is found.

# 7. Decision Surface
![Decision Surface](https://elvinouyang.github.io/assets/images/Introduction%20to%20Machine%20Learning%20with%20Python%20-%20Chapter%202%20-%20Datasets%20and%20kNN_files/Introduction%20to%20Machine%20Learning%20with%20Python%20-%20Chapter%202%20-%20Datasets%20and%20kNN_31_1.png)
    
    - Curve that seperates positive & negaive points are the decision surfaces
    - when k=1, non smooth curve, no errors & over fit
    - when k=9, smooth curve, minimal errors & well fit
    - when k = 100, smooth curve, more error & under fit
    - The smoothness of the decision surfaces increases as k increases

# 8. Overfit & underfit

![Overfit & underfit](https://upload.wikimedia.org/wikipedia/commons/1/19/Overfitting.svg)

[Source](https://upload.wikimedia.org/wikipedia/commons/1/19/Overfitting.svg)

Overfit: Extrem non-smooth decision surface so that no mistake happen in training data atleast

Underfit: Doesn't care about decision surface

Well fit: Balance overfit and underfit, less prone to noise and is robost

## How to make sure we are not overfitting and underfitting ?

![right "K"](https://datacadamia.com/_media/data_mining/knn_error_rate_best_k.jpg?w=400&h=310&buster=1391632980&tok=e10a1f)
As k increases, training error increases

- Compute train & test error and plot the error as above image. Pick the right k where train & test error converge.

    ```
    - When train & test error are high --> Underfit
    - When train error is small and test error is large --> Overfit
    ```

# 9. How to determine right "K" value in K-NN ?

- The value of k affects the k-NN classifier drastically. 
- The flexibility of the model decreases with the increase of ‘k’. With lower value of ‘k’ variance is high and bias is low but as we increase the value of ‘k’ variance starts decreasing and bias starts increasing. 
- With very low values of ‘k’ there is a chance of algorithm overfitting the data whereas with very high value of ‘k’ there is a chance of underfitting.

## K value can be determined by:
- Train KNN model with multiple k value, choose the k value that produces best accuracy. (time consuming)
- Cross validation 
- Error versus K curve

# 10. Can KNN do Multiclass classification ?

- Yup

# 11. Can KNN predict in probabilistic prediction ?

- Whoa! Yeah

# 12. KNN when dataset is imbalanced

- The prediction is biased by the majority classes.

## How to handle imbalanced dataset ?

- Under sampling
- Over sampling
- Provide more wieghtage to minority classes.

## Problem with imbalanced dataset even after handling !!

- Possibility of getting high accuracy in imbalanced dataset with dumb model in place because in test dataset (after splitting) majority classes shall be present

# 13. KNN when dataset has outliers

- when k is small (ex: k=1), KNN is prone to outlier
- perform k-fold cross validation and choose the best accuracy for the right k

# 14. KNN affected by scale of the features ?

- Yes, since KNN is distance based model.
- Hence, perform column standardization before model training

# 15. Are KNN interpretable ?

- Depends on domain
- When k is small (ex: 1), model is interpretable.
- As dimensionality increases in the dataset, interpretability becomes harder

# 16. KNN Techniques to reduce Space & Time complexity

## 16.1. kd-tree

![App Screenshot](https://www.researchgate.net/profile/Ruben-Gonzalez-Crespo/publication/327289160/figure/fig4/AS:666062085435401@1535812984438/Visualization-of-the-k-d-tree-algorithm.png)
- Break the 2D space using axis-parallel lines/planes into cuboid/rectangle/huper-cuboid
- Once the tree is formed , it is easy for algorithm to search for the probable nearest neighbor just by traversing the tree. 

### Limitations of kd-tree

- Works fine for smaller dimensionality of Datasets
- As dimensionality increases regions in kd-tree increases due to which time complexity increases
- Gives probable nearest neighbors but can miss out actual nearest neighbors

## 16.2. Locality sensitive hashing

## 16.3. Ball Tree

These are very efficient specially in case of higher dimensions.

Ball Tree are formed by following steps:

- Two clusters are created initially
- All the data points must belong to atleast one of the clusters.
- One point cannot be in both clusters.
- Distance of the point is calculated from the centroid of the each cluster. The point closer to the centroid goes into that particular cluster.
- Each cluster is then divided into sub clusters again, and then the points are classified into each cluster on the basis of distance from centroid.
- This is how the clusters are kept to be divided till a certain depth.
- Ball tree formation initially takes a lot of time but once the nested clusters are created, finding nearest neighbors is easier.

# 17. Lazy Learner

- k-NN algorithms are often termed as Lazy learners. Let’s understand why is that. 
- Most of the algorithms like Bayesian classification, logistic regression, SVM etc., are called Eager learners. 
- These algorithms generalize over the training set before receiving the test data i.e. they create a model based on the training data before receiving the test data and then do the prediction/classification on the test data. 
- But this is not the case with the k-NN algorithm. 
- It doesn’t create a generalized model for the training set but waits for the test data. 
- Once test data is provided then only it starts generalizing the training data to classify the test data. 
- So, a lazy learner just stores the training data and waits for the test set. 
- Such algorithms work less while training and more while classifying a given test dataset.

# 18. Weighted Nearest Neighbor
- In weighted k-NN, we assign weights to the k nearest neighbors.
- The weights are typically assigned on the basis of distance. 
- Sometimes rest of data points are assigned a weight of 0. 
- The main intuition is that the points in neighbor should have more weights than father points.

# 19. Pros of KNN
- It can be used for both regression and classification problems.
- It is very simple and easy to implement.
- Mathematics behind the algorithm is easy to understand.
- There is no need to create model or do hyperparameter tuning.
- KNN doesn't make any assumption for the distribution of the given data.
- There is not much time cost in training phase.

# 20. Acknowledgements

 - [Google Images](https://www.google.co.in/imghp?hl=en-GB&tab=ri&authuser=0&ogbl)
  
# 21. License

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)

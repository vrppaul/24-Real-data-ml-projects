# Importing libraries
import pandas as pd
import numpy as np
import math
import operator
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import sys
sys.path.append('../hand_made_stuff')
from handmadestuff import *

# Converting dataset into dataframe
iris = datasets.load_iris()
iris_df = pd.DataFrame(np.c_[iris.data, iris.target], 
                       columns = iris.feature_names + ['targets'])

# Splitting the dataset
X_train, X_test, y_train, y_test = hand_made_train_test_split(iris_df.iloc[:, :4], 
                                                              iris_df.iloc[:, 4])

# Testing the efficiency of the hand-made classifier
classifier = HandMadeKNN()
classifier.fit(X_train, y_train, 3)
handmade_y_pred = classifier.predict(X_test)
handmade_cm = hand_made_cm(handmade_y_pred, y_test)

# Testing the efficiency of the sklearn classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5,
                                  metric = 'minkowski',
                                  p = 2)
classifier.fit(X_train, y_train)

sklearn_y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
sklearn_cm = confusion_matrix(y_test, sklearn_y_pred)

print("""\n\n\tHandmade KNN model efficiency: {0}
      \tSklearn KNN model efficiency: {1}\n\n"""
      .format(calculate_efficiency(handmade_cm), calculate_efficiency(sklearn_cm)))






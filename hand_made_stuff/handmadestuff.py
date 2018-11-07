""" Library with different hand-made functions and classes.
Made for learning-purposes"""

# Importing libraries
import pandas as pd
import numpy as np
import math
import operator
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Defining a function which calculates euclidean distance between two data points
def euclidian_distance(point1, point2):
    """This is a nand-made function for calculating Euclidian Distance.
    Made for learning purposes."""
    
    if len(point1) != len(point2):
        raise ValueError('Points should have the same number of dimensions.')
    
    distance = 0
    dims = len(point1)
    
    for i in range(dims):
        distance += np.square(point1[i] - point2[i])
        
    return np.sqrt(distance)

# Defining the hand-made KNN class
class HandMadeKNN:
    """This is a toy hand-made class made for learning purposes.
    It's only goal is to classify and label the data put into it.
    
    'brute' is the only method available at the moment"""
    
    def __init__(self, X_train = None, y_train = None, k = None):
        """Method to declare all the needed inputs without using '.fit' method"""
        self.fit(X_train, y_train, k)
        
    def fit(self, X_train, y_train, k = 1):
        self.X_train = X_train
        self.y_train = y_train
        self.k = k
        
    def predict(self, to_predict = None):
        """Method for predicting the class of input data (only 'brute')"""
        
        prevailing_classes = []
        
        if isinstance(to_predict, pd.core.frame.DataFrame):
            length = list(to_predict.index)
        elif isinstance(to_predict[0], list) or isinstance(to_predict[0], np.ndarray):
            length = range(len(to_predict))
        else:
            length = [0]
        
        for i in length:
            # Calculating distances to any point
            distances = self._calculate_distances(to_predict, i)
            sorted_distances = self._sort_distances(distances)

            # Slicing the array of all distances to leave just k-nearest-neighbors
            sorted_distances = sorted_distances[0:self.k]

            # Defining the prevailing class
            prevailing_class = self._define_class(sorted_distances)
            prevailing_classes.append(prevailing_class)
        
        return prevailing_classes
    
    def _calculate_distances(self, to_predict, index = None):
        """Calculating distances from training and test datapoints"""
        
        distances = {}
        
        if isinstance(to_predict, pd.core.frame.DataFrame):
            to_predict = to_predict.loc[index]
        elif isinstance(to_predict[0], list) or isinstance(to_predict[0], np.ndarray):
            to_predict = to_predict[index]
    
        for i in list(self.X_train.index):

            dist = euclidian_distance(self.X_train.loc[i], to_predict)

            distances[i] = dist
            
        return distances
    
    def _sort_distances(self, dict_of_distances):
        """Sorts the dict of distances in descending order, 
        unpacks the resulted list of tuples and returns a list of sorted indexes"""
        
        sorted_distances = sorted(dict_of_distances.items(), key=lambda kv: kv[1])
        
        for i in range(len(sorted_distances)):
            sorted_distances[i] = sorted_distances[i][0]
        
        return sorted_distances
    
    def _define_class(self, sorted_distances):
        """Defining the prevailng class of nearest neighbors"""
        
        prevailng_class = (self.y_train[sorted_distances]
                            .value_counts()
                            .index
                            .values[0])
        
        return prevailng_class
    
def hand_made_cm(y_pred, y_actual):
    """Hand-made confusion matrix, 
    which recieves two parametres as input: predicted classes and real classes
    and which returns a matrix of number in a form
    
    \t\t\tpredicted values
    actual values"""
    
    if isinstance(y_pred, pd.core.series.Series):
        y_pred = y_pred.values
    if isinstance(y_actual, pd.core.series.Series):
        y_actual = y_actual.values
    
    if len(y_pred) != len(y_actual):
        raise ValueError('Amount of predicted and actual values have to be the same (len(y_pred) == len(y_actual))')
        
    n_classes = sorted(set(y_pred).union(set(y_actual)))
    size = len(sorted(set(y_pred).union(set(y_actual))))
    cm_matrix = np.zeros((size, size))
    
    arr_length = range(len(y_pred))
    
    for i in arr_length:
        if y_actual[i] == y_pred[i]:
            index = list(n_classes).index(y_actual[i])
            cm_matrix[index][index] += 1
        else:
            actual_index = list(n_classes).index(y_actual[i])
            predicted_index = list(n_classes).index(y_pred[i])
            cm_matrix[predicted_index][actual_index] += 1
    
    return(cm_matrix)
    
import random
def hand_made_train_test_split(X_values, y_values = None, test_size = 0.2):
    """Hand-made function, which splits data into train and test sets
    with a given or default percent of test samples"""
    
    set_of_indexes = set(range(len(X_values)))
    
    test_indexes = sorted(random.sample(range(len(X_values)), int(len(X_values) * test_size)))
    training_indexes = list(set_of_indexes.difference(set(test_indexes)))
    
    if y_values is None:
        return [X_values.iloc[training_indexes], X_values.iloc[test_indexes]]
    else:
         return [X_values.iloc[training_indexes], X_values.iloc[test_indexes],
                 y_values.iloc[training_indexes], y_values.iloc[test_indexes]]
         
def calculate_efficiency(cm):
    """Function which calculates efficiency of the given confusion matrix"""
    
    sum_total = 0
    sum_predicted = 0
    
    for i in range(len(cm)):
        sum_total += sum(cm[i])
        sum_predicted += cm[i][i]
        
    return sum_predicted / sum_total

def visualise_classifier(dataset_name, classifier, X_train, y_train):
    """Function that contains all steps of the code which visualises the classifier's results"""
    
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    
    plt.figure(figsize = (10,10)) 
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green', 'blue'))(i),
                    label = dataset_name.target_names[int(j)], s = 50)
    plt.xlabel('\nPrincipal Component 1', fontsize = 15)
    plt.ylabel('Principal Component 2', fontsize = 15)
    plt.title('\n2 component PCA\n', fontsize = 20)
    plt.legend()
    plt.show()
    
def fit_predict_report(dataset_name, classifier_name, X_train, y_train, X_test, y_test, 
                       n_neighbors = 5, svc_kernel = 'rbf', svc_c = 1,
                      visualise = False):
    """Function that calculates precision of the chosen classifier
    
    If visualise == True, a plot of given classifier for given data is plotted"""
    if classifier_name == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = n_neighbors)
    elif classifier_name == 'svc':
        from sklearn.svm import SVC
        classifier = SVC(kernel = svc_kernel, C = svc_c)
    elif classifier_name == 'lr':
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression()
     
    # Fitting classifier to the Training set
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Printing the name of the classifier
    print("Classificator used:\n {}\n\n".format(classifier_name))
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix: \n\n', cm, '\n\n')
    
    # Displaying reports
    from sklearn.metrics import classification_report
    print('Classification report:\n\n', classification_report(y_test, y_pred))
    
    if visualise:
        visualise_classifier(dataset_name, classifier, X_train, y_train)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SequentialFeatureSelector
import numpy as np


def train(data, k):
    """ 
    input:
        data: the first column is labels, remaining columns are features
        k: the parameter for k-nearest neighbor model
    output:
        the knn model trained on data
    """
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(data.loc[:, 1:16], data.loc[:, 0])
    return knn_model


def eval(data, model):
    """ 
    input: 
        data: the first column is labels, remaining columns are features
        model: knn model to be tested on the data
    output: 
        the score of the model on the testing dataset
    """
    return model.score(data.loc[:, 1:16], data.loc[:, 0])


def tune_k(data, list=range(1, 10)):
    """ 
    input: 
        data: the first column is labels, remaining columns are features
        list: list of k-values to be considered
    output: 
        the k-value in the list that gives the best model
    """
    scores = np.zeros(len(list))
    for i in list:
        model = KNeighborsClassifier(n_neighbors=i)
        scores[i] = np.mean(cross_val_score(model, data.loc[:, 1:16], data.loc[:, 0]))
    return list[np.argmax(scores)]


def tune_metric(data):
    pass


def dimensionReduction(data, k=3):
    """ 
    input: 
        data: the first column is labels, remaining columns are features
        k: optional parameter for the knn model
    output: 
        a reduced dataset that has the labels in the first column, and 4 selected features
    """
    knn_model = KNeighborsClassifier(n_neighbors=k)
    sfs = SequentialFeatureSelector(knn_model, n_features_to_select=4)
    sfs.fit(data.loc[:, 1:16], data.loc[:, 0])
    return data.loc[:, np.insert(sfs.get_support(), 0, 1)]

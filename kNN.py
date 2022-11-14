from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SequentialFeatureSelector
import numpy as np


def train(data, k):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(data.loc[:, 1:16], data.loc[:, 0])
    return knn_model


def eval(data, model):
    return model.score(data.loc[:, 1:16], data.loc[:, 0])


def tune_k(data, list=range(1, 10)):
    scores = np.zeros(len(list))
    for i in list:
        model = KNeighborsClassifier(n_neighbors=i)
        scores[i] = np.mean(cross_val_score(model, data.loc[:, 1:16], data.loc[:, 0]))
    return list[np.argmax(scores)]


def tune_metric(data):
    pass


def dimensionReduction(data, k=3):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    sfs = SequentialFeatureSelector(knn_model, n_features_to_select=4)
    sfs.fit(data.loc[:, 1:16], data.loc[:, 0])
    return data.loc[:, np.insert(sfs.get_support(), 0, 1)]

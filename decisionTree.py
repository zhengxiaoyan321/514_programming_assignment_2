import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


def train(data, criterion="gini", max_depth=None):
    """
    input:
        data: the first column is labels, remaining columns are features
        criterion: The function to measure the quality of a split.
        max_depth: The maximum depth of the tree.
    output:
        the k-value in the list that gives the best model
    """
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=123)
    clf.fit(data.iloc[:, 1:17], data.iloc[:, 0])
    return clf

def eval(model, data):
    """
    input:
        data: the first column is labels, remaining columns are features
        model: knn model to be tested on the data
    output:
        the score of the model on the testing dataset
    """
    return model.score(data.iloc[:, 1:17], data.iloc[:, 0])

def tune_criterion(data, list):
    """
    input:
        data: the first column is labels, remaining columns are features
        list: list of parameters to be considered
    output:
        the parameter in the list that gives the best model
    """
    scores = []
    for i in list:
        model = DecisionTreeClassifier(criterion=i, random_state=123)
        scores.append(np.mean(cross_val_score(model, data.iloc[:, 1:17], data.iloc[:, 0])))
    # return list[np.argmax(scores)]
    return scores


def tune_max_depth(data, list):
    """
    input:
        data: the first column is labels, remaining columns are features
        list: list of parameters to be considered
    output:
        the cross validation scores
    """
    scores = []
    for i in list:
        model = DecisionTreeClassifier(max_depth=i, random_state=123)
        scores.append(np.mean(cross_val_score(model, data.iloc[:, 1:17], data.iloc[:, 0])))
    # return list[np.argmax(scores)]
    return scores


def dimensionReduction(data, criterion="gini"):
    """
    input:
        data: the first column is labels, remaining columns are features
        k: optional parameter for the knn model
    output:
        a reduced dataset that has the labels in the first column, and 4 selected features
    """
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=4, random_state=123)
    clf.fit(data.loc[:, 1:17], data.loc[:, 0])
    a = clf.feature_importances_
    selected_features = sorted(range(len(a)), key=lambda i: a[i])[-4:]
    return data.loc[:, np.insert(selected_features, 0, 0)]
    # return data.loc[:, selected_features]

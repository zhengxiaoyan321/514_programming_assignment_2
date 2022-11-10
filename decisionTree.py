import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

def decisionTree(data):
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(data.loc[:,1:16], data.loc[:,0])
    return clf

def tune(data, list):
    scores = np.zeros(len(list))
    for i in list:
        model = DecisionTreeClassifier()
        scores[i] = np.mean(cross_val_score(model, data.loc[:, 1:16], data.loc[:, 0]))
    return list[np.argmax(scores)]
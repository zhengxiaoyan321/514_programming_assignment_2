import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def preprocessing():
    """ 
    This function read the dataset and select relevant samples for the binary classification problems. For each pair, 10% samples are set aside as final validation set.
    Output:
        three pairs of training, testing dataset for three binary classification problems
    """
    # read data and select relative pairs for classification
    full_data = pd.read_csv("data\letter-recognition.data", header=None)
    HK = full_data.loc[(full_data[0] == "H") | (full_data[0] == "K"), :]
    MY = full_data.loc[(full_data[0] == "M") | (full_data[0] == "Y"), :]
    OQ = full_data.loc[(full_data[0] == "O") | (full_data[0] == "Q"), :]

    # set aside 10% samples for each pair
    [HK_train, HK_test] = train_test_split(HK, test_size=0.1, random_state=12345)
    [MY_train, MY_test] = train_test_split(MY, test_size=0.1, random_state=12345)
    [OQ_train, OQ_test] = train_test_split(OQ, test_size=0.1, random_state=12345)

    return [[HK_train, HK_test], [MY_train, MY_test], [OQ_train, OQ_test]]

def multi_class_preprocessing():
    """ 
    This function read the dataset and select relevant samples for the multi-class classification problems. 10% samples are set aside as final validation set.
    Output:
        a training dataset and a testing dataset each containing 6 classes
    """
    full_data = pd.read_csv("data\letter-recognition.data", header=None)
    data = full_data.loc[full_data[0].str.fullmatch('[H,K,M,Y,O,Q]'), :]
    [train, test] = train_test_split(data, test_size=0.1, random_state=12345)
    return [train, test]

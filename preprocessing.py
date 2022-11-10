import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def preprocessing():
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

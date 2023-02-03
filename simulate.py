import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
 
import load_data
from preprocessing import StandardScale, LabelEncoder, OnehotEncoder
from train import StartifiedCV

import pandas as pd
import numpy as np

from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier

datapath = 'open'
train, test = load_data.load_data(datapath)

train_X, train_y = load_data.split_data_label(train)
test, _ = load_data.split_data_label(test, False)

def preprocess(train_X, train_y, test):
    train_X = train_X.fillna(0)
    test = test.fillna(0)
    train_X, test, line_dum = OnehotEncoder.preprocess(train_X, test, ['LINE'])
    train_X, test, prod_dum = OnehotEncoder.preprocess(train_X, test, ['PRODUCT_CODE'])
    train_X, test = StandardScale.preprocess(train_X, test, [x for x in train_X.columns if 'X_' in x])

    return train_X, train_y, test

def training(train_X, train_y):
    model = KNeighborsClassifier(n_neighbors=3)
    models, scores = StartifiedCV.train(train_X, train_y, model, n_splits=5)

    return models, scores

def predict(models, test):
    preds = np.array([])
    for m in models:
        preds = np.append(preds,m.predict(test))
    preds = mode(preds.reshape((-1,5)), axis=1).mode.reshape(-1)
    return preds

def submission(preds, basepath, filename):
    submit = pd.read_csv('{}/sample_submission.csv'.format(basepath))
    submit['Y_Class'] = preds
    submit.to_csv('{}.csv'.format(filename), index=False)

train_X, train_y, test = preprocess(train_X, train_y, test)
models, scores = training(train_X, train_y)
preds = predict(models, test)
submission(preds, datapath, 'KNN')
import random
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


import load_data
from preprocessing import OnehotEncoder, RemoveEmptyColumn, RemoveStdZeroColumn, ConcatProdLine, DataScaling
import pandas as pd
import numpy as np

from imblearn.over_sampling import BorderlineSMOTE

from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

seed = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def preprocess(train_X, train_y, test):
    train_X, test = RemoveEmptyColumn.preprocess(train_X, test)
    train_X, test = RemoveStdZeroColumn.preprocess(train_X, test)
    train_X, test, new_col = ConcatProdLine.preprocess(train_X, test)
    train_X = train_X.fillna(0)
    test = test.fillna(0)
    train_X, test = DataScaling.robust(train_X, test)
    train_X, test, prod_dum = OnehotEncoder.preprocess(train_X, test, [new_col])

    return train_X, train_y, test

def training(train_X, train_y, test):
    models = [
        LGBMClassifier(objective='multiclass', random_state=seed),
        CatBoostClassifier(objective='MultiClass',
                            task_type='GPU',
                            one_hot_max_size=2, random_seed=42,
                            iterations=4000, early_stopping_rounds=50,
                            learning_rate=0.05, verbose=False
                            ),
        XGBClassifier(random_state=seed),
    ]
    
    [x.fit(train_X, train_y) for x in models]
    
    return models

def predict(models, test, mode, weights):
    if mode == "hard":
        preds = np.asarray([x.predict(test).reshape(-1) for x in models]).T
        res = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x, weights=weights)),
            axis=1,
            arr=preds
        )
    if mode == "soft":
        preds = np.asarray([x.predict_proba(test) for x in models])
        res = np.zeros(preds[0].shape)
        for pred, weight in zip(preds, weights):
            res = res + pred*weight
        res = np.argmax(preds, axis=0)
    return res

def submission(preds, basepath, filename):
    """
    
    """
    submit = pd.read_csv('{}/sample_submission.csv'.format(basepath))
    submit['Y_Class'] = preds
    submit.to_csv('{}.csv'.format(filename), index=False)

def main():
    seed_everything(42) # Seed 고정
    datapath = r'C:/Users/mooha/Desktop/LG Aimers/open'
    train, test = load_data.load_data(datapath)

    train_X, train_y = load_data.split_data_label(train)
    test, _ = load_data.split_data_label(test, False)

    train_X, train_y, test = preprocess(train_X, train_y, test)
    
    models = training(train_X, train_y, test)
    preds = predict(models, test, "hard", [2,2,1])
    submission(preds, datapath, 'hard')


if __name__ == "__main__": 
    main()
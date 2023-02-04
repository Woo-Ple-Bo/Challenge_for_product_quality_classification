import random
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
 
import load_data
from preprocessing import StandardScale,LabelEncoder, OnehotEncoder, RemoveEmptyColumn, RemoveStdZeroColumn
from train import StratifiedCV, Train_test_split

import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
from scipy.stats import mode
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def preprocess(train_X, train_y, test):
    train_X, test = RemoveEmptyColumn.preprocess(train_X, test)
    train_X, test = RemoveStdZeroColumn.preprocess(train_X, test)
    train_X = train_X.fillna(0)
    test = test.fillna(0)
    train_X, test, line_dum = OnehotEncoder.preprocess(train_X, test, ['LINE'])
    train_X, test, prod_dum = OnehotEncoder.preprocess(train_X, test, ['PRODUCT_CODE'])
    train_X, test = StandardScale.preprocess(train_X, test, [x for x in train_X.columns if 'X_' in x])

    return train_X, train_y, test

def training(train_X, train_y):
    model = XGBClassifier(random_state=42)
    models, scores = Train_test_split.train(train_X, train_y, model)
    return models, scores

def predict(models, test):
    preds = np.array([])
    for m in models:
        preds = np.append(preds,m.predict(test))
    preds = mode(preds.reshape((-1,len(models))), axis=1).mode.reshape(-1)
    return preds

def submission(preds, basepath, filename):
    submit = pd.read_csv('{}/sample_submission.csv'.format(basepath))
    submit['Y_Class'] = preds
    submit.to_csv('{}.csv'.format(filename), index=False)

def compare(train_X, train_y):
    models = {
        'lgbm' : LGBMClassifier(objective='multiclass', random_state=42),
        'cat' : CatBoostClassifier(objective='MultiClass',task_type='GPU'),
        'knn' : KNeighborsClassifier(n_neighbors=3),
        'xgb' : XGBClassifier(random_state=42),
        'ada' : AdaBoostClassifier(),
        }
    trains = {
        'train_test':Train_test_split.train,
        'stratifiedCV':StratifiedCV.train
    }
    res = ''
    for model_name, model in models.items():
        for train_name, train in trains.items():
            models, scores = train(train_X, train_y, model)
            res += '{}_{}\n'.format(model_name, train_name)
            res += '{}\n'.format(scores)
            res += '{}\n'.format(np.mean(scores))
            res += '\n'
    print(res)

def main():
    seed_everything(42) # Seed 고정
    datapath = 'open'
    train, test = load_data.load_data(datapath)

    train_X, train_y = load_data.split_data_label(train)
    test, _ = load_data.split_data_label(test, False)

    train_X, train_y, test = preprocess(train_X, train_y, test)
    #compare(train_X, train_y)
    models, scores = training(train_X, train_y)
    print(scores)
    print('score : ', np.mean(scores))
    preds = predict(models, test)
    submission(preds, datapath, 'xgb')

if __name__ == "__main__": 
    main()
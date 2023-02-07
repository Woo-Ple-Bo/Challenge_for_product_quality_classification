import random
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
 
import load_data
from preprocessing import StandardScale,LabelEncoder, OnehotEncoder, RemoveEmptyColumn, RemoveStdZeroColumn, ConcatProdLine
from train import StratifiedCV, Train_test_split, VanillaCV

import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
from scipy.stats import mode
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, VotingClassifier,RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

seed = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def preprocess(train_X, train_y, test):
    train_X, test, new_col = ConcatProdLine.preprocess(train_X, test)
    train_X, test = RemoveEmptyColumn.preprocess(train_X, test)
    train_X, test = RemoveStdZeroColumn.preprocess(train_X, test)
    train_X = train_X.fillna(0)
    test = test.fillna(0)
    #train_X, test, line_dum = OnehotEncoder.preprocess(train_X, test, ['LINE'])
    #train_X, test, prod_dum = OnehotEncoder.preprocess(train_X, test, ['PRODUCT_CODE'])
    train_X, test, prod_dum = OnehotEncoder.preprocess(train_X, test, [new_col])
    train_X, test = StandardScale.preprocess(train_X, test, [x for x in train_X.columns if 'X_' in x])

    return train_X, train_y, test

def training(train_X, train_y):
    model = CatBoostClassifier(objective='MultiClass',
                                        task_type='GPU',
                                        one_hot_max_size=2, random_seed=42,
                                        iterations=4000, early_stopping_rounds=500,
                                        learning_rate=0.05
                                        )
    model.fit(train_X, train_y)
    models = [model]
    scores = [0]
    return models, scores

def predict(models, test):
    preds = np.array([])
    for m in models:
        preds = np.append(preds,m.predict(test))
    preds = mode(preds.reshape((-1,len(models))), axis=1).mode.reshape(-1, 1)
    return preds

def submission(preds, basepath, filename):
    submit = pd.read_csv('{}/sample_submission.csv'.format(basepath))
    submit['Y_Class'] = preds
    submit.to_csv('{}.csv'.format(filename), index=False)

def compare(train_X, train_y):    
    models = {
        'cat2' : {
            'model':CatBoostClassifier(objective='MultiClass',
                                        task_type='GPU',
                                        one_hot_max_size=4, random_seed=42,
                                        iterations=4000, early_stopping_rounds=300,
                                        learning_rate=0.05
                                        ),
            'fit_params' : {'verbose': 200},
        },
        'cat4' : {
            'model':CatBoostClassifier(objective='MultiClass',
                                        task_type='GPU',
                                        one_hot_max_size=2, random_seed=42,
                                        iterations=4000, early_stopping_rounds=300,
                                        learning_rate=0.05
                                        ),
            'fit_params' : {'verbose': 200},
        },
    }
    trains = {
        'train_test':Train_test_split.train,
        #'stratifiedCV':StratifiedCV.train,
        #'vanillaCV':VanillaCV.train
    }

    res = ''
    for model_name, model_n_params in models.items():
        for train_name, train in trains.items():
            model = model_n_params['model']
            params = model_n_params['fit_params']
            models, scores = train(train_X, train_y, model, params)
            res += '{}_{}\n'.format(model_name, train_name)
            res += '{}\n'.format(np.mean(scores))
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
    submission(preds, datapath, 'tuned_cat')

if __name__ == "__main__": 
    main()
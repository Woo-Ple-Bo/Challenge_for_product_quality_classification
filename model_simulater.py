# base modules
import random
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# custom modules 
import load_data
from preprocessing import StandardScale,LabelEncoder, OnehotEncoder, RemoveEmptyColumn, RemoveStdZeroColumn, ConcatProdLine, RemoveOverNA, LeaveDupColumns, NumericToStrForOnehot
from train import StratifiedCV, Train_test_split, VanillaCV
# data processing modules
import pandas as pd
import numpy as np
from imblearn.over_sampling import BorderlineSMOTE
# evaluate modules
from sklearn.metrics import f1_score
from scipy.stats import mode
# classifiers
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
    #train_X, test = RemoveOverNA.preprocess(train_X, test)
    #train_X, test = LeaveDupColumns.preprocess(train_X, test)
    train_X, test = RemoveEmptyColumn.preprocess(train_X, test)
    train_X, test = RemoveStdZeroColumn.preprocess(train_X, test)
    train_X, test, new_col = ConcatProdLine.preprocess(train_X, test)
    train_X = train_X.fillna(0)
    test = test.fillna(0)
    #train_X, test = NumericToStrForOnehot.preprocess(train_X, test, 10)
    #train_X, test, line_dum = OnehotEncoder.preprocess(train_X, test, ['LINE'])
    #train_X, test, prod_dum = OnehotEncoder.preprocess(train_X, test, ['PRODUCT_CODE'])
    train_X, test, prod_dum = OnehotEncoder.preprocess(train_X, test, [new_col])
    train_X, train_y = BorderlineSMOTE(random_state=42).fit_resample(train_X, train_y)
    #train_X, test = StandardScale.preprocess(train_X, test, [x for x in train_X.columns if 'X_' in x])

    return train_X, train_y, test

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

def compare(train_X, train_y):
    models = {
        #'cat' : {
        #    'model':CatBoostClassifier(objective='MultiClass',
        #                                task_type='GPU',
        #                                one_hot_max_size=2, random_seed=42,
        #                                iterations=4000, verbose=False,
        #                                learning_rate=0.05
        #                                ),
        #},
        'lgbm' : {
            'model':LGBMClassifier(objective='multiclass', random_state=seed),
            },
        'xgb' : {
            'model':XGBClassifier(random_state=seed),
            },
        'knn' : {
            'model':KNeighborsClassifier(n_neighbors=3),
            },
        
        'ada' : {
            'model':AdaBoostClassifier(),
        },
        'bag' : {
            'model':BaggingClassifier(random_state=seed),
        },
        'dt' : {
            'model':DecisionTreeClassifier(random_state=seed),
        },
        'rc' : {
            'model':RidgeClassifier(random_state=seed),
        },
        'gb' : {
            'model':GradientBoostingClassifier(random_state=seed),
        },
        'svc' : {
            'model':SVC(random_state=seed),
        },
        'rcc' : {
            'model':RidgeClassifierCV(),
        },
        'rf' : {
            'model':RandomForestClassifier(random_state=seed),
        },
    }
    
    models['soft_vote'] = {
        'model' :VotingClassifier([
            ('lgbm', LGBMClassifier(objective='multiclass', random_state=seed)),
            ('xgb', XGBClassifier(random_state=seed)),
            ('knn', KNeighborsClassifier(n_neighbors=3)),
            ('ada', AdaBoostClassifier()),
            ('bag', BaggingClassifier(random_state=seed)),
            ('dt', DecisionTreeClassifier(random_state=seed)),
            ('gb', GradientBoostingClassifier(random_state=seed)),
            ('svc', SVC(random_state=seed, probability=True)),
            ('rf', RandomForestClassifier(random_state=seed)),
            ], voting='soft')
        }
    
    trains = {
        'train_test':Train_test_split.train,
    }
    report = pd.DataFrame(columns=trains.keys())
    for model_name, model_n_params in models.items():
        tmp = []
        for train_name, train in trains.items():
            print(model_name)
            model = model_n_params['model']
            if 'fit_params' in model_n_params.keys():
                params = model_n_params['fit_params']
            else:
                params = None
            model, scores = train(train_X, train_y, model, params)
            tmp.append(scores)
        report.loc[model_name]=tmp

    report.index = list(models.keys())
    report.to_csv('report.csv')

def main():
    seed_everything(42) # Seed 고정
    datapath = 'open'
    train, test = load_data.load_data(datapath)

    train_X, train_y = load_data.split_data_label(train)
    test, _ = load_data.split_data_label(test, False)

    train_X, train_y, test = preprocess(train_X, train_y, test)
    
    compare(train_X, train_y)

if __name__ == "__main__": 
    main()
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
import pandas as pd

def preprocess(train_X, test):
    T05 = train_X[train_X['LINE']=='T010305'].dropna(axis=1, how='all')
    T06 = train_X[train_X['LINE']=='T010306'].dropna(axis=1, how='all')
    T04 = train_X[train_X['LINE']=='T050304'].dropna(axis=1, how='all')
    T07 = train_X[train_X['LINE']=='T050307'].dropna(axis=1, how='all')
    T14 = train_X[train_X['LINE']=='T100304'].dropna(axis=1, how='all')
    T16 = train_X[train_X['LINE']=='T100306'].dropna(axis=1, how='all')

    A_union = list((set(T05)|set(T06)|set(T04)|set(T07))-(set(['LINE','PRODUCT_CODE'])))
    NA_union = list((set(T14)|set(T16))-(set(['LINE','PRODUCT_CODE'])))

    imp_mean = IterativeImputer(estimator = LinearRegression(), 
                       tol= 1e-10, 
                       max_iter=5, 
                       verbose=2, 
                       imputation_order='roman')
    imp_mean.fit(train_X[A_union])
    train_X[A_union] = imp_mean.transform(train_X[A_union])
    test[A_union] = imp_mean.transform(test[A_union])

    imp_mean = IterativeImputer()
    imp_mean.fit(train_X[NA_union])
    train_X[NA_union] = imp_mean.transform(train_X[NA_union])
    test[NA_union] = imp_mean.transform(test[NA_union])

    return train_X, test
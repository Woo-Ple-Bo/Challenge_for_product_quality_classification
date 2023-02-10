import pandas as pd

def preprocess(train_X, test, n_uni):
    for col in train_X.columns:
        if len(train_X[col].unique()) < n_uni:
            train_X[col] = train_X[col].fillna(0).astype('str')
            test[col] = test[col].fillna(0).astype('str')
    return train_X, test
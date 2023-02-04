import pandas as pd

def load_data(path):
    train = pd.read_csv('{}/train.csv'.format(path))
    test = pd.read_csv('{}/test.csv'.format(path))
    return train, test

def split_data_label(df, is_train=True):
    if is_train:
        x = df.drop(['PRODUCT_ID','Y_Class', 'Y_Quality','TIMESTAMP'], axis = 1)
        y = df['Y_Class']
        return x,y
    else:
        x = df.drop(['PRODUCT_ID','TIMESTAMP'], axis = 1)
        return x, None
    
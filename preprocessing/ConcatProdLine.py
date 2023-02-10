import pandas as pd


def preprocess(train_X, test):
    """ PROD_LINE과 LINE 컬럼 병합

    Parameters
    ----------
    train_X : df.DataFrame
        학습할 데이터셋의 데이터프레임

    test : df.DataFrame
        추론할 데이터셋의 데이터프레임

    Returns
    -------
    train_X : df.DataFrame
        인코딩된 학습 데이터프레임
    test : df.DataFrame
        인코딩된 추론 데이터프레임
    
    Notes
    -----
    PRODUCT_CODE 열과 LINE 열을 병합
    """

    train_X['PROD_LINE'] = train_X['PRODUCT_CODE']+'_'+train_X['LINE']
    train_X = train_X.drop(['PRODUCT_CODE','LINE'],axis=1)
    test['PROD_LINE'] = test['PRODUCT_CODE']+'_'+test['LINE']
    test = test.drop(['PRODUCT_CODE','LINE'],axis=1)

    return train_X, test, ['PROD_LINE']
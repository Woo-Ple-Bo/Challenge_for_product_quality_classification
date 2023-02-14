import pandas as pd

def preprocess(train_X, test):
    """ 샘플이 적거나 결측치가 많은 열을 제거

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
    의미없는 컬럼을 제거
    """
    
    train_X = train_X.loc[:,~train_X.T.duplicated(keep='first')]
    test = test[train_X.columns]

    return train_X, test
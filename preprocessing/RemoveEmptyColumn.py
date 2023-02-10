import pandas as pd


def preprocess(train_X, test):
    """어떤 열의 모든 값이 null인 열을 제거

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
    이 함수는 모든행이 na인 열을 제거
    """

    train_X = train_X.dropna(axis=1, how='all')
    test = test[train_X.columns]

    return train_X, test
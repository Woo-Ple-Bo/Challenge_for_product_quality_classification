import pandas as pd


def preprocess(train_X, test):
    """표준편차 0인 열 제거

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
    이 함수는 열의 표준편차가 0인 열을 제거
    """

    for col in [x for x in train_X.columns if 'X_' in x]:
        if len(train_X[col].value_counts())==1:
            train_X = train_X.drop(col, axis=1)
        
    test = test[train_X.columns]

    return train_X, test
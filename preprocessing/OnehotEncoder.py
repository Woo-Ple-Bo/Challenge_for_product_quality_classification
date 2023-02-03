import pandas as pd

def preprocess(train_X, test, target_col):
    """원핫 인코딩

    Parameters
    ----------
    train_X : df.DataFrame
        학습할 데이터셋의 데이터프레임

    test : df.DataFrame
        추론할 데이터셋의 데이터프레임
    
    target_col : list
        인코딩할 컬럼리스트

    Returns
    -------
    train_X : df.DataFrame
        인코딩된 학습 데이터프레임
    test : df.DataFrame
        인코딩된 추론 데이터프레임
    dummies_col : list
        추가된 컬럼리스트

    Notes
    -----
    이 함수는 인코딩이후 생성된 컬럼리스트를 추가로 반환\n
    ex) A -> 1 0 0 0
        B -> 0 1 0 0
    """
    dummies_col = []
    for c in target_col:
        df = pd.get_dummies(train_X[c])
        train_X[df.columns] = df
        train_X = train_X.drop(c, axis=1)
        df = pd.get_dummies(test[c])
        test[df.columns] = df
        test = test.drop(c, axis=1)
        dummies_col.extend(df.columns)
    return train_X, test, dummies_col
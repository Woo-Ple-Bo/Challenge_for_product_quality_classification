from sklearn.preprocessing import MaxAbsScaler


def get_values(value):
    return value.values.reshape(-1, 1)

def preprocess(train_X, test, target_col):
    """원핫 인코딩

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
    이 함수는 target_col의 컬럼들을 표준화한다
    """
    for col in target_col:
        scaler = MaxAbsScaler()
        train_X[col] = scaler.fit_transform(get_values(train_X[col]))
        if col in test.columns:
            test[col] = scaler.transform(get_values(test[col]))
    return train_X, test
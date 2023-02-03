from sklearn.preprocessing import LabelEncoder

def get_values(value):
    return value.values.reshape(-1, 1)

def preprocess(train_X, test, target_col):
    """라벨 인코딩

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
        
    Notes
    -----
    이 함수는 target_col의 문자형 컬럼을 수치형으로 변환\n
    ex) A -> 1, B-> 2
    """
    le = LabelEncoder()
    for col in target_col:    
        train_X[col] = le.fit_transform(train_X[col])
        if col in test.columns:
            test[col] = le.transform(test[col])
    return train_X, test
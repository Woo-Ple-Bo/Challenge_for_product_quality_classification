import pandas as pd

def preprocess(train_X, test):
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
    line_uni = train_X['LINE'].unique()
    a_cols = []
    for line in [x for x in line_uni if x.startswith('T0')]:
        a_cols.append(train_X[train_X['LINE']==line].dropna(axis=1, how='all').columns)
    a_dup_cols = set(train_X.columns)
    for col in a_cols:
        a_dup_cols = a_dup_cols&set(col)

    na_cols = []
    for line in [x for x in line_uni if x.startswith('T1')]:
        na_cols.append(train_X[train_X['LINE']==line].dropna(axis=1, how='all').columns)
    na_dup_cols = set(train_X.columns)
    for col in na_cols:
        na_dup_cols = na_dup_cols&set(col)
    
    del_cols = set(a_dup_cols) | set(na_dup_cols)
    total_cols = list(set(train_X.columns) - del_cols)
    total_cols.append('LINE')
    total_cols.append('PRODUCT_CODE')
    train_X = train_X[total_cols]
    test = test[total_cols]

    return train_X, test
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer

def standard(train_x, test):
    """
    StandardSclaer입니다.
    특성들의 평균을 0, 분산을 1로 스케일링 하는 것입니다.

    최솟값과 최댓값의 크기를 제한하지 않기 때문에, 어떤 알고리즘에서는 문제가 있을 수 있으며
    이상치에 매우 민감합니다.
    회귀보다 분류에 유용합니다.
    """
    scaler = StandardScaler()

    num_features_train = [x for x in train_x.columns if "X" in x]

    train_x[num_features_train] = scaler.fit_transform(train_x[num_features_train])
    test[num_features_train] = scaler.transform(test[num_features_train])

    return train_x, test

def minmax(train_x, test):
    """
    MinMaxScaler입니다.
    가장 작은 값은 0, 가장 큰 값은 1로 변환되므로, 모든 특성들은 [0, 1] 범위를 갖게 됩니다.

    이상치에 매우 민감합니다.
    분류보다 회귀에 유용합니다.
    """
    scaler = MinMaxScaler()

    num_features_train = [x for x in train_x.columns if "X" in x]

    train_x[num_features_train] = scaler.fit_transform(train_x[num_features_train])
    test[num_features_train] = scaler.transform(test[num_features_train])

    return train_x, test

def maxabs(train_x, test):
    """
    MaxAbsScaler입니다.
    각 특성의 절대값이 0 과 1 사이가 되도록 스케일링합니다.
    모든 값은 -1 과 1 사이로 표현되며, 데이터가 양수일 경우 MinMaxScaler 와 같습니다.
    이상치에 매우 민감합니다.
    """
    scaler = MaxAbsScaler()

    num_features_train = [x for x in train_x.columns if "X" in x]

    train_x[num_features_train] = scaler.fit_transform(train_x[num_features_train])
    test[num_features_train] = scaler.transform(test[num_features_train])

    return train_x, test

def robust(train_x, test):
    """
    RobustScaler 입니다.
    평균과 분산 대신에 중간 값과 사분위 값을 사용합니다.
    중간 값은 정렬시 중간에 있는 값을 의미하고
    사분위값은 1/4, 3/4에 위치한 값을 의미합니다.
    이상치 영향을 최소화할 수 있습니다.
    """
    scaler = RobustScaler()

    num_features_train = [x for x in train_x.columns if "X" in x]
    #train_x.select_dtypes(exclude=['object']).columns.to_list()

    train_x[num_features_train] = scaler.fit_transform(train_x[num_features_train])
    test[num_features_train] = scaler.transform(test[num_features_train])

    return train_x, test

def normalize(train_x, test):
    """
    Normalizer 입니다.
    각 열(특성)의 통계치를 이용하여 진행하지않고
    각 행(샘플)마다 적용되는 방식입니다.
    일반적인 데이터 전처리의 상황에서 사용되는 것이 아니라
    모델(특히 딥러닝)내 학습 벡터에 적용하며,
    특히나 피쳐들이 다른 단위(키, 나이, 소득 등)라면 더더욱 사용하지 않습니다.
    """
    scaler = Normalizer()

    num_features_train = [x for x in train_x.columns if "X" in x]

    train_x[num_features_train] = scaler.fit_transform(train_x[num_features_train])
    test[num_features_train] = scaler.transform(test[num_features_train])

    return train_x, test
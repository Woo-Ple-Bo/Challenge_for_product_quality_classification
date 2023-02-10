from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

def train(train_X, train_y, model, params):
    models = []
    scores = []
    skf = KFold(n_splits=5)
    skf.get_n_splits(train_X, train_y)
    print(params)
    for i, (train_index, test_index) in enumerate(skf.split(train_X, train_y)):
        if params:
            model.fit(train_X.iloc[train_index], train_y.iloc[train_index], **params)
        else:
            model.fit(train_X.iloc[train_index], train_y.iloc[train_index])
        pred = model.predict(train_X.iloc[test_index])
        score = f1_score(train_y.reset_index(drop=True)[test_index], pred, average='macro')
        models.append(model)
        scores.append(score)

    return models, scores
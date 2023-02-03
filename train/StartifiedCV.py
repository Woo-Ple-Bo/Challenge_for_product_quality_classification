from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

def train(train_X, train_y, model, n_splits=5):
    models = []
    scores = []
    skf = StratifiedKFold(n_splits)
    skf.get_n_splits(train_X, train_y)
    for i, (train_index, test_index) in enumerate(skf.split(train_X, train_y)):
        model.fit(train_X.iloc[train_index], train_y.iloc[train_index])
        pred = model.predict(train_X.iloc[test_index])
        score = f1_score(train_y[test_index], pred, average='macro')
        models.append(model)
        scores.append(score)

    return models, scores
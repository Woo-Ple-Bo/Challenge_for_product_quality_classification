from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

def train(train_X, train_y, model, params):
    models = []
    scores = []
    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(train_X, train_y)
    for i, (train_index, test_index) in enumerate(skf.split(train_X, train_y)):
        val_y = train_y.reset_index(drop=True)[test_index]
        if params:
            model.fit(train_X.iloc[train_index], train_y.iloc[train_index], **params)
        else:
            model.fit(train_X.iloc[train_index], train_y.iloc[train_index])
        pred = model.predict(train_X.iloc[test_index])
        score = f1_score(val_y, pred, average='macro')
        models.append(model)
        scores.append(classification_report(val_y, pred))
    return models, scores
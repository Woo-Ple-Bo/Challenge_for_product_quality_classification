import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

def train(train_X, train_y, model, params):
    models = []
    scores = []

    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2)
    if params:
        model.fit(train_X, train_y, **params)
    else:
        model.fit(train_X, train_y)
    pred = model.predict(val_X)
    score = f1_score(val_y, pred, average='macro')
    models.append(model)
    scores.append(classification_report(val_y, pred))
    return models, scores
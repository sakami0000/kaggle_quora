import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold

from config import seed


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def threshold_search(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.001)
    f = 2 / (1 / precision + 1 / recall)
    best_th = thresholds[np.argmax(f)]
    return best_th


def scoring(y_true, y_proba):
    rkf = RepeatedStratifiedKFold(random_state=seed)

    scores = []
    ths = []
    for train_index, test_index in rkf.split(y_true, y_true):
        y_prob_train, y_prob_test = y_proba[train_index], y_proba[test_index]
        y_true_train, y_true_test = y_true[train_index], y_true[test_index]

        # determine best threshold on 'train' part
        best_threshold = threshold_search(y_true_train, y_prob_train)

        # use this threshold on 'test' part for score
        sc = f1_score(y_true_test, (y_prob_test >= best_threshold).astype(int))
        scores.append(sc)
        ths.append(best_threshold)

    best_th = np.mean(ths)
    score = np.mean(scores)

    search_result = {'threshold': best_th, 'score': score}
    return search_result

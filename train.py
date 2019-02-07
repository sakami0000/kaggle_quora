import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from config import seed
from preprocessing import NBFeaturer, tokenize
from utils import threshold_search


def train(max_iter=40, n_splits=20):
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')

    # TF-IDF feature
    tfidf = TfidfVectorizer(
        ngram_range=(1, 4),
        tokenizer=tokenize,
        min_df=3,
        max_df=0.9,
        strip_accents='unicode',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    ).fit(pd.concat([train['question_text'], test['question_text']]))

    train_x = tfidf.transform(train['question_text'])
    test_x = tfidf.transform(test['question_text'])
    train_y = train['target'].values

    # Naive Bayes scaling
    nb_transformer = NBFeaturer(alpha=1).fit(train_x, train_y)
    train_nb = nb_transformer.transform(train_x)
    test_nb = nb_transformer.transform(test_x)

    # train
    models = []
    train_meta = np.zeros(train_y.shape)
    test_meta = np.zeros(test_x.shape[0])

    splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(train, train_y))

    for idx, (train_idx, valid_idx) in enumerate(splits):
        x_train_fold = train_nb[train_idx]
        y_train_fold = train_y[train_idx]

        x_val_fold = train_nb[valid_idx]
        y_val_fold = train_y[valid_idx]

        model = LogisticRegression(solver='lbfgs', dual=False, class_weight='balanced', C=0.5, max_iter=max_iter)
        model.fit(x_train_fold, y_train_fold)
        models.append(model)

        valid_pred = model.predict_proba(x_val_fold)
        train_meta[valid_idx] = valid_pred[:, 1]
        test_meta += model.predict_proba(test_nb)[:, 1] / len(splits)

    # search threshold
    best_th = threshold_search(train_y, train_meta)

    # submit
    sub = pd.read_csv('../input/sample_submission.csv')
    sub.prediction = test_meta > best_th
    sub.to_csv('submission.csv', index=False)

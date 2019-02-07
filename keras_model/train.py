import numpy as np
import pandas as pd

from load_data import load_and_prec, load_glove, load_para
from utils import threshold_search


def train_pred(model, epochs=2):
    # load data
    train_x, val_x, test_x, train_y, val_y, word_index = load_and_prec()
    embedding_matrix_1 = load_glove(word_index)
    embedding_matrix_2 = load_para(word_index)
    embedding_matrix = np.mean([embedding_matrix_1, embedding_matrix_2], axis=0)

    # train
    model = model(embedding_matrix)
    model.fit(train_x, train_y, batch_size=512, epochs=epochs, validation_data=(val_x, val_y))
    pred_val_y = model.predict([val_x], batch_size=1024, verbose=0)
    pred_test_y = model.predict([test_x], batch_size=1024, verbose=0)

    # search threshold
    best_th = threshold_search(val_y, pred_val_y)

    # submit
    sub = pd.read_csv('../input/sample_submission.csv')
    sub.prediction = pred_test_y > best_th
    sub.to_csv('submission.csv', index=False)

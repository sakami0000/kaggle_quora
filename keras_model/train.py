import numpy as np
import pandas as pd

from load_data import load_and_prec, load_glove, load_para
from utils import threshold_search


def train_keras(train, valid, test, embedding_matrix, model, epochs=6, batch_size=512):

    train_x, train_y = train
    val_x, val_y = valid
    (test_x,) = test

    # train
    model = model(embedding_matrix)
    model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_data=(val_x, val_y))
    pred_val_y = model.predict([val_x], batch_size=batch_size, verbose=0)
    pred_test_y = model.predict([test_x], batch_size=batch_size, verbose=0)

    # search threshold
    best_th = threshold_search(val_y, pred_val_y)

    # predict
    preds = (pred_test_y > best_th).astype(int)

    return preds

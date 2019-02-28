import gc

import numpy as np
import pandas as pd

from load_data import load_and_prec, load_and_prec_with_val, load_glove, load_para
from logistic_regression.train import train_logreg
from keras_model.train import train_keras
from torch_model.train import train_torch


def main_torch(neuralnet, train_epochs=5, n_splits=5, batch_size=512):

    # load data
    train, test, word_index, embeddings_index = load_and_prec()
    embedding_matrix_1 = load_glove(word_index, embeddings_index)
    embedding_matrix_2 = load_para(word_index)

    embedding_matrix = embedding_matrix_1 * 0.6 + embedding_matrix_2 * 0.4

    del embeddings_index, embedding_matrix_1, embedding_matrix_2
    gc.collect()

    # train & predict
    preds = train_torch(train, test, embedding_matrix, neuralnet,
                        train_epochs=train_epochs, n_splits=n_splits, batch_size=batch_size)

    # submit
    sub = pd.read_csv('../input/sample_submission.csv')
    sub.prediction = preds
    sub.to_csv('submission.csv', index=False)


def main_keras(model, epochs=6, batch_size=512):

    # load data
    train, valid, test, word_index = load_and_prec_with_val()
    embedding_matrix_1 = load_glove(word_index)
    embedding_matrix_2 = load_para(word_index)
    embedding_matrix = np.mean([embedding_matrix_1, embedding_matrix_2], axis=0)

    # train & predict
    preds = train_keras(train, valid, test, embedding_matrix, model,
                        epochs=epochs, batch_size=batch_size)

    # submit
    sub = pd.read_csv('../input/sample_submission.csv')
    sub.prediction = preds
    sub.to_csv('submission.csv', index=False)


def main_logreg(max_iter=40, n_splits=20):

    # load data
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')

    # train & predict
    preds = train_logreg(train, test, max_iter=max_iter, n_splits=n_splits)

    # submit
    sub = pd.read_csv('../input/sample_submission.csv')
    sub.prediction = preds
    sub.to_csv('submission.csv', index=False)

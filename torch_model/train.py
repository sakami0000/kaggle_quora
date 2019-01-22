import gc
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch

from config import seed, seed_torch
from load_data import load_and_prec, load_glove, load_para
from torch_model.callbacks import CyclicLR
from utils import cos_annealing_lr, scoring, sigmoid

warnings.filterwarnings('ignore')


def train(neuralnet, train_epochs=5, n_splits=5, batch_size=512):

    train_x, test_x, train_y, features, test_features, word_index, embeddings_index = load_and_prec()
    embedding_matrix_1 = load_glove(embeddings_index, word_index)
    embedding_matrix_2 = load_para(word_index)

    embedding_matrix = embedding_matrix_1 * 0.6 + embedding_matrix_2 * 0.4

    del embeddings_index, embedding_matrix_1, embedding_matrix_2
    gc.collect()

    train_preds = np.zeros((len(train_x)))
    test_preds = np.zeros((len(test_x)))

    seed_torch(seed)

    x_test_cuda = torch.tensor(test_x, dtype=torch.long).cuda()
    test = torch.utils.data.TensorDataset(x_test_cuda)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(train_x, train_y))

    for fold, (train_idx, valid_idx) in enumerate(splits):
        x_train_fold = torch.tensor(train_x[train_idx], dtype=torch.long).cuda()
        y_train_fold = torch.tensor(train_y[train_idx, np.newaxis], dtype=torch.float32).cuda()
        f_train_fold = features[train_idx]

        x_val_fold = torch.tensor(train_x[valid_idx], dtype=torch.long).cuda()
        y_val_fold = torch.tensor(train_y[valid_idx, np.newaxis], dtype=torch.float32).cuda()
        f_val_fold = features[valid_idx]

        model = neuralnet(embedding_matrix)
        model.cuda()

        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
        step_size = 300
        base_lr, max_lr = 0.001, 0.003
        optimizer = torch.optim.Adam(model.parameters())

        scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                             step_size=step_size, mode='exp_range',
                             gamma=0.9994)

        train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
        valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

        print(f'Fold {fold + 1}')

        for epoch in range(train_epochs):
            start_time = time.time()

            model.train()
            avg_loss = 0.

            for i, (x_batch, y_batch) in enumerate(train_loader):
                f = f_train_fold[i * batch_size:(i + 1) * batch_size]
                y_pred = model([x_batch, f])

                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)

            model.eval()
            valid_preds_fold = np.zeros((x_val_fold.size(0)))
            test_preds_fold = np.zeros(len(test_x))
            avg_val_loss = 0.

            for i, (x_batch, y_batch) in enumerate(valid_loader):
                f = f_val_fold[i * batch_size:(i + 1) * batch_size]
                y_pred = model([x_batch, f]).detach()

                if scheduler:
                    scheduler.batch_step()

                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                valid_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

            elapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, train_epochs, avg_loss, avg_val_loss, elapsed_time))

        for i, (x_batch,) in enumerate(test_loader):
            f = test_features[i * batch_size:(i + 1) * batch_size]
            y_pred = model([x_batch, f]).detach()

            test_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

        train_preds[valid_idx] = valid_preds_fold
        test_preds += test_preds_fold / len(splits)

    search_result = scoring(train_y, train_preds)

    sub = pd.read_csv('../input/sample_submission.csv')
    sub.prediction = test_preds > search_result['threshold']
    sub.to_csv('submission.csv', index=False)
import gc
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch

from config import seed
from load_data import load_and_prec, load_and_prec_with_len, load_glove, load_para
from torch_model.callbacks import CyclicLR
from torch_model.utils import seed_torch, SentenceLengthDataset
from utils import cos_annealing_lr, scoring, sigmoid

warnings.filterwarnings('ignore')


def train_torch(neuralnet, train_epochs=5, n_splits=5, batch_size=512):

    # load data
    train_x, test_x, train_y, features, test_features, word_index, embeddings_index = load_and_prec()
    embedding_matrix_1 = load_glove(embeddings_index, word_index)
    embedding_matrix_2 = load_para(word_index)

    embedding_matrix = embedding_matrix_1 * 0.6 + embedding_matrix_2 * 0.4

    del embeddings_index, embedding_matrix_1, embedding_matrix_2
    gc.collect()

    # train
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

        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
        step_size = 300
        base_lr, max_lr = 0.001, 0.003
        optimizer = torch.optim.Adam(model.parameters())

        scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                             step_size=step_size, mode='exp_range',
                             gamma=0.99994)

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
                with torch.no_grad():
                    y_pred = model([x_batch, f]).detach()

                scheduler.batch_step()

                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                valid_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

            elapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, train_epochs, avg_loss, avg_val_loss, elapsed_time))

        for i, (x_batch,) in enumerate(test_loader):
            f = test_features[i * batch_size:(i + 1) * batch_size]
            with torch.no_grad():
                y_pred = model([x_batch, f]).detach()

            test_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

        train_preds[valid_idx] = valid_preds_fold
        test_preds += test_preds_fold / len(splits)

    # search threshold
    search_result = scoring(train_y, train_preds)

    # submit
    sub = pd.read_csv('../input/sample_submission.csv')
    sub.prediction = test_preds > search_result['threshold']
    sub.to_csv('submission.csv', index=False)


def snapshot_train(neuralnet, n_cycle=2, epochs_per_cycle=5, n_splits=4, batch_size=1024):

    # load data
    train_x, test_x, train_y, features, test_features, word_index, embeddings_index = load_and_prec()
    embedding_matrix_1 = load_glove(embeddings_index, word_index)
    embedding_matrix_2 = load_para(word_index)

    embedding_matrix = embedding_matrix_1 * 0.6 + embedding_matrix_2 * 0.4

    del embeddings_index, embedding_matrix_1, embedding_matrix_2
    gc.collect()

    # train
    train_preds = np.zeros((len(train_x)))
    test_preds = np.zeros((len(test_x)))

    seed_torch(seed)

    initial_lr = 0.1

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

        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
        optimizer = torch.optim.Adam(model.parameters())

        train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
        valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

        print(f'\nFold {fold + 1}')

        for cycle in range(n_cycle):

            print(f'Cycle {cycle + 1}')

            for epoch in range(epochs_per_cycle):

                start_time = time.time()

                lr = cos_annealing_lr(initial_lr, epoch, epochs_per_cycle)
                optimizer.state_dict()['param_groups'][0]['lr'] = lr

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
                    with torch.no_grad():
                        y_pred = model([x_batch, f]).detach()

                    avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                    valid_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

                elapsed_time = time.time() - start_time
                print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                    epoch + 1, epochs_per_cycle, avg_loss, avg_val_loss, elapsed_time))

            for i, (x_batch,) in enumerate(test_loader):
                f = test_features[i * batch_size:(i + 1) * batch_size]
                with torch.no_grad():
                    y_pred = model([x_batch, f]).detach()

                test_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

            train_preds[valid_idx] += valid_preds_fold / n_cycle
            test_preds += test_preds_fold / (len(splits) * n_cycle)

    # search threshold
    search_result = scoring(train_y, train_preds)

    # submit
    sub = pd.read_csv('../input/sample_submission.csv')
    sub.prediction = test_preds > search_result['threshold']
    sub.to_csv('submission.csv', index=False)


def packed_train(neuralnet, train_epochs=5, n_splits=5, batch_size=512):

    # load data
    train_x, test_x, train_y, features, test_features,\
        train_len, test_len, word_index, embeddings_index = load_and_prec_with_len()
    embedding_matrix_1 = load_glove(embeddings_index, word_index)
    embedding_matrix_2 = load_para(word_index)

    embedding_matrix = embedding_matrix_1 * 0.6 + embedding_matrix_2 * 0.4

    del embeddings_index, embedding_matrix_1, embedding_matrix_2
    gc.collect()

    # train
    train_preds = np.zeros((len(train_x)))
    test_preds = np.zeros((len(test_x)))

    seed_torch(seed)

    x_test_cuda = torch.tensor(test_x, dtype=torch.long).cuda()
    test = torch.utils.data.TensorDataset(x_test_cuda)
    test = SentenceLengthDataset(test, test_len, test_features)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(train_x, train_y))

    for fold, (train_idx, valid_idx) in enumerate(splits):
        x_train_fold = torch.tensor(train_x[train_idx], dtype=torch.long).cuda()
        y_train_fold = torch.tensor(train_y[train_idx, np.newaxis], dtype=torch.float32).cuda()
        f_train_fold = features[train_idx]
        l_train_fold = train_len[train_idx]

        x_val_fold = torch.tensor(train_x[valid_idx], dtype=torch.long).cuda()
        y_val_fold = torch.tensor(train_y[valid_idx, np.newaxis], dtype=torch.float32).cuda()
        f_val_fold = features[valid_idx]
        l_val_fold = train_len[valid_idx]

        model = neuralnet(embedding_matrix)
        model.cuda()

        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
        step_size = 300
        base_lr, max_lr = 0.001, 0.003
        optimizer = torch.optim.Adam(model.parameters())

        scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                             step_size=step_size, mode='exp_range',
                             gamma=0.99994)

        train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
        valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

        train = SentenceLengthDataset(train, l_train_fold, f_train_fold)
        valid = SentenceLengthDataset(valid, l_val_fold, f_val_fold)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

        print(f'Fold {fold + 1}')

        for epoch in range(train_epochs):
            start_time = time.time()

            model.train()
            avg_loss = 0.

            for i, ((x_batch, y_batch), l, f) in enumerate(train_loader):
                y_pred = model([x_batch, f], l=l)

                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)

            model.eval()
            valid_preds_fold = np.zeros((x_val_fold.size(0)))
            test_preds_fold = np.zeros(len(test_x))
            avg_val_loss = 0.

            for i, ((x_batch, y_batch), l, f) in enumerate(valid_loader):
                with torch.no_grad():
                    y_pred = model([x_batch, f], l=l).detach()

                scheduler.batch_step()

                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                valid_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

            elapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, train_epochs, avg_loss, avg_val_loss, elapsed_time))

        for i, ((x_batch,), l, f) in enumerate(test_loader):
            with torch.no_grad():
                y_pred = model([x_batch, f], l=l).detach()

            test_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

        train_preds[valid_idx] = valid_preds_fold
        test_preds += test_preds_fold / len(splits)

    # search threshold
    search_result = scoring(train_y, train_preds)

    # submit
    sub = pd.read_csv('../input/sample_submission.csv')
    sub.prediction = test_preds > search_result['threshold']
    sub.to_csv('submission.csv', index=False)

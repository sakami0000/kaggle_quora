from contextlib import contextmanager
import re
import time
import gc
import random
import os
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
import torch.utils.data
from torch.optim.optimizer import Optimizer

embed_size = 300
max_features = 95000
maxlen = 72

batch_size = 1536
train_epochs = 6
n_splits = 7

SEED = 1029


@contextmanager
def timer(msg):
    t0 = time.time()
    print(f'[{msg}] start.')
    yield
    elapsed_time = time.time() - t0
    print(f'[{msg}] done in {elapsed_time / 60:.2f} min.')


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']',
          '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^',
          '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶',
          '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼',
          '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
          'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪',
          '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']


def clean_text(x):
    x = str(x)
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, f' {punct} ')
    return x


def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x


def add_features(embeddings_index, df):
    df['num_words'] = df.question_text.str.count('\S+')
    df['oov'] = df['question_text'].apply(
        lambda comment: sum(1 for w in comment.split() if embeddings_index.get(w) is None))
    df['oov_vs_words'] = df['oov'] / df['num_words']

    return df


def load_and_prec():
    train_df = pd.read_csv('../input/train.csv')
    test_df = pd.read_csv('../input/test.csv')
    print('Train shape : ', train_df.shape)
    print('Test shape : ', test_df.shape)

    # lower
    train_df['question_text'] = train_df['question_text'].str.lower()
    test_df['question_text'] = test_df['question_text'].str.lower()

    # Clean the text
    train_df['question_text'] = train_df['question_text'].apply(clean_text)
    test_df['question_text'] = test_df['question_text'].apply(clean_text)

    # Clean numbers
    train_df['question_text'] = train_df['question_text'].apply(clean_numbers)
    test_df['question_text'] = test_df['question_text'].apply(clean_numbers)

    # fill up the missing values
    train_X = train_df['question_text'].fillna('_##_').values
    test_X = test_df['question_text'].fillna('_##_').values

    # load embedding
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(' ')) for o in open(EMBEDDING_FILE))

    # add features
    train = add_features(embeddings_index, train_df)
    test = add_features(embeddings_index, test_df)

    features = train[['oov_vs_words']].fillna(0).astype('float64')
    test_features = test[['oov_vs_words']].fillna(0).astype('float64')

    ss = StandardScaler()
    ss.fit(np.vstack((features, test_features)))
    features = ss.transform(features)
    test_features = ss.transform(test_features)

    # Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    # Pad the sentences
    train_X = pad_sequences(train_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    # Get the target values
    train_y = train_df['target'].values

    # shuffling the data
    np.random.seed(SEED)
    trn_idx = np.random.permutation(len(train_X))

    train_X = train_X[trn_idx]
    train_y = train_y[trn_idx]
    features = features[trn_idx]

    return train_X, test_X, train_y, features, test_features, tokenizer.word_index, embeddings_index


def load_glove(embeddings_index, word_index):
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = -0.005838499, 0.48782197
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = embeddings_index.get(word.capitalize())
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return embedding_matrix


def load_para(word_index):
    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    emb_mean, emb_std = -0.0053247833, 0.49346462

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    with open(EMBEDDING_FILE, 'r', encoding='utf8', errors='ignore') as f:
        for line in f:
            if len(line) > 100:
                word, vec = line.split(' ', 1)
                if word not in word_index:
                    continue
                i = word_index[word]
                if i >= max_features:
                    continue
                embedding_vector = np.asarray(vec.split(' '), dtype='float32')[:300]
                if len(embedding_vector) == 300:
                    embedding_matrix[i] = embedding_vector
    return embedding_matrix


class CyclicLR(object):
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError('expected {} base_lr, got {}'.format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError('expected {} max_lr, got {}'.format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma ** (x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        hidden_size = 60

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(0.1)

        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(hidden_size * 2, maxlen)
        self.gru_attention = Attention(hidden_size * 2, maxlen)

        self.linear = nn.Linear(hidden_size * 8 + 1, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(16)
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        h_embedding = self.embedding(x[0])
        h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))

        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)

        h_lstm_atten = self.lstm_attention(h_lstm)
        h_gru_atten = self.gru_attention(h_gru)

        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)

        f = torch.tensor(x[1], dtype=torch.float).cuda()

        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool, f), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        conc = self.bn(conc)
        out = self.out(conc)

        return out


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def scoring(y_true, y_proba):

    def threshold_search(y_true, y_proba):
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        thresholds = np.append(thresholds, 1.001)
        F = 2 / (1 / precision + 1 / recall)
        best_th = thresholds[np.argmax(F)]
        return best_th

    rkf = RepeatedStratifiedKFold(random_state=SEED)

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


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    with timer('load data'):
        train_X, test_X, train_y, features, test_features, word_index, embeddings_index = load_and_prec()
        embedding_matrix_1 = load_glove(embeddings_index, word_index)
        embedding_matrix_2 = load_para(word_index)

        embedding_matrix = embedding_matrix_1 * 0.6 + embedding_matrix_2 * 0.4

        del embeddings_index, embedding_matrix_1, embedding_matrix_2
        gc.collect()

    with timer('train'):
        train_preds = np.zeros((len(train_X)))
        test_preds = np.zeros((len(test_X)))

        seed_torch(SEED)

        x_test_cuda = torch.tensor(test_X, dtype=torch.long).cuda()
        test = torch.utils.data.TensorDataset(x_test_cuda)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

        splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(train_X, train_y))

        for fold, (train_idx, valid_idx) in enumerate(splits):
            x_train_fold = torch.tensor(train_X[train_idx], dtype=torch.long).cuda()
            y_train_fold = torch.tensor(train_y[train_idx, np.newaxis], dtype=torch.float32).cuda()
            f_train_fold = features[train_idx]

            x_val_fold = torch.tensor(train_X[valid_idx], dtype=torch.long).cuda()
            y_val_fold = torch.tensor(train_y[valid_idx, np.newaxis], dtype=torch.float32).cuda()
            f_val_fold = features[valid_idx]

            model = NeuralNet()
            model.cuda()

            loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
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
                test_preds_fold = np.zeros(len(test_X))
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

    with timer('submit'):
        warnings.filterwarnings('ignore')
        search_result = scoring(train_y, train_preds)
        print('CV score: {:.4f}'.format(search_result['score']))

        sub = pd.read_csv('../input/sample_submission.csv')
        sub.prediction = test_preds > search_result['threshold']
        sub.to_csv('submission.csv', index=False)

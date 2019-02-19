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
import torch.nn.functional as F
import torch.utils.data
from torch.optim.optimizer import Optimizer

embed_size = 300
max_features = 120000
maxlen = 70

batch_size = 512
train_epochs = 5
n_splits = 5

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


puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/',
          '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', '·', '√',
          '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥',
          '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–', '‡',
          '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═',
          '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪',
          '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è',
          '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪',
          '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤']


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


mispell_dict = {"aren't": "are not", "can't": "cannot", "couldn't": "could not",
                "didn't": "did not", "doesn't": "does not", "don't": "do not",
                "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                "he'd": "he would", "he'll": "he will", "he's": "he is",
                "i'd": "I had", "i'll": "I will", "i'm": "I am", "isn't": "is not",
                "it's": "it is", "it'll": "it will", "i've": "I have", "let's": "let us",
                "mightn't": "might not", "mustn't": "must not", "shan't": "shall not",
                "she'd": "she would", "she'll": "she will", "she's": "she is",
                "shouldn't": "should not", "that's": "that is", "there's": "there is",
                "they'd": "they would", "they'll": "they will", "they're": "they are",
                "they've": "they have", "we'd": "we would", "we're": "we are",
                "weren't": "were not", "we've": "we have", "what'll": "what will",
                "what're": "what are", "what's": "what is", "what've": "what have",
                "where's": "where is", "who'd": "who would", "who'll": "who will",
                "who're": "who are", "who's": "who is", "who've": "who have",
                "won't": "will not", "wouldn't": "would not", "you'd": "you would",
                "you'll": "you will", "you're": "you are", "you've": "you have",
                "'re": " are", "wasn't": "was not", "we'll": " will", "tryin'": "trying",
                'colour': 'color', 'centre': 'center', 'favourite': 'favorite',
                'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater',
                'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization',
                'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ',
                'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist',
                'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can',
                'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I',
                'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation',
                'mastrubate': 'masturbate', 'mastrubating': 'masturbating', 'pennis': 'penis',
                'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data',
                '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend',
                'airhostess': 'air hostess', 'whst': 'what', 'watsapp': 'whatsapp',
                'demonitisation': 'demonetization', 'demonitization': 'demonetization',
                'demonetisation': 'demonetization'}


def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispellings, mispellings_re = _get_mispell(mispell_dict)


def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)


def add_features(df):
    df['question_text'] = df['question_text'].apply(lambda x: str(x))
    df['total_length'] = df['question_text'].apply(len)
    df['capitals'] = df['question_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals']) / float(row['total_length']), axis=1)

    df['num_words'] = df.question_text.str.count('\S+')
    df['num_unique_words'] = df['question_text'].apply(lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']

    return df


def load_and_prec():
    train_df = pd.read_csv('../input/train.csv')
    test_df = pd.read_csv('../input/test.csv')
    print('Train shape : ', train_df.shape)
    print('Test shape : ', test_df.shape)

    # clean the text
    train_df['question_text'] = train_df['question_text'].apply(clean_text)
    test_df['question_text'] = test_df['question_text'].apply(clean_text)

    # clean numbers
    train_df['question_text'] = train_df['question_text'].apply(clean_numbers)
    test_df['question_text'] = test_df['question_text'].apply(clean_numbers)

    # clean speelings
    train_df['question_text'] = train_df['question_text'].apply(replace_typical_misspell)
    test_df['question_text'] = test_df['question_text'].apply(replace_typical_misspell)

    # fill up the missing values
    train_X = train_df['question_text'].fillna('_##_').values
    test_X = test_df['question_text'].fillna('_##_').values

    # add features
    train = add_features(train_df)
    test = add_features(test_df)

    features = train[['caps_vs_length', 'words_vs_unique']].fillna(0)
    test_features = test[['caps_vs_length', 'words_vs_unique']].fillna(0)

    ss = StandardScaler()
    ss.fit(np.vstack((features, test_features)))
    features = ss.transform(features)
    test_features = ss.transform(test_features)

    # tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    # pad the sentences
    train_X = pad_sequences(train_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    # get the target values
    train_y = train_df['target'].values

    # shuffling the data
    np.random.seed(SEED)
    trn_idx = np.random.permutation(len(train_X))

    train_X = train_X[trn_idx]
    train_y = train_y[trn_idx]
    features = features[trn_idx]

    return train_X, test_X, train_y, features, test_features, tokenizer.word_index


def load_glove(word_index):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(' ')) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
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

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(
        get_coefs(*o.split(' ')) for o in open(EMBEDDING_FILE, encoding='utf8', errors='ignore') if len(o) > 100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
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


class CapsLayer(nn.Module):
    def __init__(self, input_dim_capsule=120, num_capsule=5, dim_capsule=16,
                 routings=4, kernel_size=(9, 1), share_weights=True,
                 activation='default', batch_size=512, **kwargs):
        super(CapsLayer, self).__init__(**kwargs)

        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights

        self.t_epsilon = 1e-7
        if activation == 'default':
            self.activation = self.squash
        else:
            self.activation = nn.ReLU(inplace=True)

        if self.share_weights:
            self.W = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(1, input_dim_capsule, self.num_capsule * self.dim_capsule)))
        else:
            self.W = nn.Parameter(
                torch.randn(batch_size, input_dim_capsule, self.num_capsule * self.dim_capsule))

    def forward(self, x):

        if self.share_weights:
            u_hat_vecs = torch.matmul(x, self.W)
        else:
            print('add later')

        batch_size = x.size(0)
        input_num_capsule = x.size(1)
        u_hat_vecs = u_hat_vecs.view((batch_size, input_num_capsule,
                                      self.num_capsule, self.dim_capsule))
        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3)
        b = torch.zeros_like(u_hat_vecs[:, :, :, 0])

        for i in range(self.routings):
            b = b.permute(0, 2, 1)
            c = F.softmax(b, dim=2)
            c = c.permute(0, 2, 1)
            b = b.permute(0, 2, 1)
            outputs = self.activation(torch.einsum('bij,bijk->bik', (c, u_hat_vecs)))
            if i < self.routings - 1:
                b = torch.einsum('bik,bijk->bij', (outputs, u_hat_vecs))
        return outputs

    def squash(self, x, axis=-1):
        s_squared_norm = (x ** 2).sum(axis, keepdim=True)
        scale = torch.sqrt(s_squared_norm + self.t_epsilon)
        return x / scale


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
        caps_out = 1

        num_capsule = 5
        dim_capsule = 5

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.1)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(hidden_size * 2, maxlen)
        self.gru_attention = Attention(hidden_size * 2, maxlen)

        self.lincaps = nn.Linear(num_capsule * dim_capsule, caps_out)
        self.caps_layer = CapsLayer(input_dim_capsule=hidden_size * 2,
                                    num_capsule=num_capsule,
                                    dim_capsule=dim_capsule,
                                    routings=4)

        self.linear = nn.Linear(hidden_size * 8 + caps_out + 2, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(16, momentum=0.5)
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

        content3 = self.caps_layer(h_gru)
        content3 = self.dropout(content3)
        batch_size = content3.size(0)
        content3 = content3.view(batch_size, -1)
        content3 = self.relu(self.lincaps(content3))

        f = torch.tensor(x[1], dtype=torch.float).cuda()

        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool, content3, f), 1)
        conc = self.relu(self.linear(conc))
        conc = self.bn(conc)
        conc = self.dropout(conc)
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

    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)

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


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    with timer('load data'):
        train_X, test_X, train_y, features, test_features, word_index = load_and_prec()
        embedding_matrix_1 = load_glove(word_index)
        embedding_matrix_2 = load_para(word_index)

        embedding_matrix = embedding_matrix_1 * 0.6 + embedding_matrix_2 * 0.4

        del embedding_matrix_1, embedding_matrix_2
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
                                 gamma=0.99994)

            train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
            valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

            train = MyDataset(train)
            valid = MyDataset(valid)

            train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

            print(f'Fold {fold + 1}')

            for epoch in range(train_epochs):
                start_time = time.time()

                model.train()
                avg_loss = 0.
                for x_batch, y_batch, index in train_loader:
                    f = f_train_fold[index]
                    y_pred = model([x_batch, f])

                    if scheduler:
                        scheduler.batch_step()

                    loss = loss_fn(y_pred, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.item() / len(train_loader)

                model.eval()
                valid_preds_fold = np.zeros((x_val_fold.size(0)))
                test_preds_fold = np.zeros(len(test_X))
                avg_val_loss = 0.
                for i, (x_batch, y_batch, index) in enumerate(valid_loader):
                    f = f_val_fold[index]
                    y_pred = model([x_batch, f]).detach()

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

    with timer('submit'):
        warnings.filterwarnings('ignore')
        search_result = scoring(train_y, train_preds)
        print('CV score: {:.4f}'.format(search_result['score']))

        sub = pd.read_csv('../input/sample_submission.csv')
        sub.prediction = test_preds > search_result['threshold']
        sub.to_csv('submission.csv', index=False)

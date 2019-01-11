from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import seed, max_features, maxlen
from preprocessing import clean_text, clean_numbers, replace_typical_misspell, add_features

EMBEDDING_GLOVE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
EMBEDDING_PARA = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_and_prec():
    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")

    # lower
    train_df["question_text"] = train_df["question_text"].apply(lambda x: x.lower())
    test_df["question_text"] = test_df["question_text"].apply(lambda x: x.lower())

    # Clean the text
    train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x))
    test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))

    # Clean numbers
    train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_numbers(x))
    test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_numbers(x))

    # Clean spellings
    train_df["question_text"] = train_df["question_text"].apply(lambda x: replace_typical_misspell(x))
    test_df["question_text"] = test_df["question_text"].apply(lambda x: replace_typical_misspell(x))

    # fill up the missing values
    train_x = train_df["question_text"].fillna("_##_").values
    test_x = test_df["question_text"].fillna("_##_").values

    # load embedding
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_GLOVE))

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
    tokenizer.fit_on_texts(list(train_x))
    train_x = tokenizer.texts_to_sequences(train_x)
    test_x = tokenizer.texts_to_sequences(test_x)

    # Pad the sentences
    train_x = pad_sequences(train_x, maxlen=maxlen)
    test_x = pad_sequences(test_x, maxlen=maxlen)

    # Get the target values
    train_y = train_df['target'].values

    # shuffling the data
    np.random.seed(seed)
    trn_idx = np.random.permutation(len(train_x))

    train_x = train_x[trn_idx]
    train_y = train_y[trn_idx]
    features = features[trn_idx]

    return train_x, test_x, train_y, features, test_features, tokenizer.word_index, embeddings_index


def load_glove(embeddings_index, word_index):
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= max_features:
            continue

        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = embeddings_index.get(word.capitalize())
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return embedding_matrix


def load_para(word_index):
    embeddings_index = dict(
        get_coefs(*o.split(" "))
        for o in open(EMBEDDING_PARA, encoding='utf8', errors='ignore')
        if len(o) > 100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= max_features:
            continue

        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = embeddings_index.get(word.capitalize())
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return embedding_matrix

from keras.initializers import glorot_normal, orthogonal
from keras.layers import (
    Input, Dense, Dropout, Embedding, Reshape, Flatten,
    Conv2D, MaxPool2D, concatenate, SpatialDropout1D,
    Bidirectional, CuDNNLSTM, CuDNNGRU,
    GlobalAveragePooling1D, GlobalMaxPooling1D,
    BatchNormalization
)
from keras.models import Model

from config import embed_size, max_features, maxlen, seed
from keras_model.layers import Attention, Capsule, CRF


def model_cnn(embedding_matrix):
    filter_sizes = [1, 2, 3, 5]
    num_filters = 36

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Reshape((maxlen, embed_size, 1))(x)

    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                      kernel_initializer='he_normal', activation='elu')(x)
        pool = MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv)
        maxpool_pool.append(pool)

    z = concatenate(axis=1)(maxpool_pool)
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def model_lstm_gru(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.1)(x)

    x = Bidirectional(CuDNNLSTM(40, return_sequences=True))(x)
    y = Bidirectional(CuDNNGRU(40, return_sequences=True))(x)

    atten_1 = Attention(maxlen)(x)
    atten_2 = Attention(maxlen)(y)
    avg_pool = GlobalAveragePooling1D()(y)
    max_pool = GlobalMaxPooling1D()(y)

    conc = concatenate([atten_1, atten_2, avg_pool, max_pool])
    conc = Dense(16, activation='relu')(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(1, activation='sigmoid')(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def model_lstm_atten(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)

    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = Attention(maxlen)(x)

    x = Dense(64, activation='relu')(x)
    outp = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def model_gru_srk_atten(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)

    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = Attention(maxlen)(x)

    x = Dense(16, activation='relu')(x)
    x = Dropout(0.1)(x)
    outp = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def model_lstm_du(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)

    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation='relu')(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(1, activation='sigmoid')(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def model_gru_atten_3(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)

    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(100, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = Attention(maxlen)(x)

    outp = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def capsule_model(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features,
                  embed_size,
                  input_length=maxlen,
                  weights=[embedding_matrix],
                  trainable=False)(inp)
    x = SpatialDropout1D(0.28)(x)

    x = Bidirectional(CuDNNGRU(128, activation='relu', dropout=0.25,
                               recurrent_dropout=0.25, return_sequences=True))(x)

    x = Capsule(num_capsule=10, dim_capsule=16,
                routings=5, share_weights=True)(x)

    x = Flatten()(x)
    x = Dropout(0.25)(x)
    outp = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def gru_crf(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(rate=0.24)(x)
    x = Bidirectional(CuDNNGRU(100, return_sequences=True,
                               kernel_initializer=glorot_normal(seed=seed),
                               recurrent_initializer=orthogonal(gain=1.0, seed=seed)))(x)

    x = CRF(10, learn_mode='marginal', unroll=True)(x)

    x = Flatten()(x)
    x = Dense(100, activation='relu', kernel_initializer=glorot_normal(seed=seed))(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    outp = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

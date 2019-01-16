from keras import initializers, regularizers, constraints
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Activation


class Attention(Layer):
    def __init__(self, step_dim,
                 w_regularizer=None, b_regularizer=None,
                 w_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.w_regularizer = regularizers.get(w_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.w_constraint = constraints.get(w_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.w = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.w_regularizer,
                                 constraint=self.w_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.w, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)
        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a

        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


class Capsule(Layer):
    def __init__(self, num_capsule=5, dim_capsule=16, routings=4, kernel_size=(9, 1),
                 share_weights=True, activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)

        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights

        if activation == 'default':
            self.activation = self.squash
        else:
            self.activation = Activation(activation)

    @staticmethod
    def squash(x, axis=-1):
        s_squashed_norm = K.sum(K.square(x), axis, keepdims=True)
        scale = K.sqrt(s_squashed_norm + K.epsilon())
        return x / scale

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]

        if self.share_weights:
            self.w = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.w = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.w)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.w, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])

        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))

            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))

            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

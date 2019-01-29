import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CapsuleLayer(nn.Module):
    def __init__(self, input_dim_capsule, num_capsule, dim_capsule,
                 routings=4, kernel_size=(9, 1), share_weights=True,
                 t_epsilon=1e-7, batch_size=1024, activation='default', **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)

        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights

        self.t_epsilon = t_epsilon

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


class BackHook(torch.nn.Module):
    def __init__(self, hook):
        super(BackHook, self).__init__()
        self._hook = hook
        self.register_backward_hook(self._backward)

    def forward(self, *inp):
        return inp

    @staticmethod
    def _backward(self, grad_in, grad_out):
        self._hook()
        return None


class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()
        self.hooker = BackHook(lambda: self._backward())

    def _setup(self):
        for name_w in self.weights:
            w = getattr(self.module, name_w)
            self.register_parameter(name_w + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self, name_w + '_raw')
            mask = raw_w.new_ones((raw_w.size(0), 1))
            mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
            w = mask.expand_as(raw_w) * raw_w
            rnn_w = getattr(self.module, name_w)
            rnn_w.data.copy_(w)
            setattr(self, name_w + '_mask', mask)

    def _backward(self):
        # transfer gradients from embeddedRNN to raw params
        for name_w in self.weights:
            raw_w = getattr(self, name_w + '_raw')
            rnn_w = getattr(self.module, name_w)
            raw_w.grad = rnn_w.grad * getattr(self, name_w + '_mask')

    def forward(self, *args):
        self._setweights()
        return self.module(*self.hooker(*args))


class CIN(nn.Module):
    def __init__(self, input_size, layer_sizes=(32, 32), activation='relu',
                 bn=False, bias=True, direct=False, **kwargs):
        if len(layer_sizes) == 0:
            raise ValueError(
                'layer_size must be a list(tuple) of length greater than 1')

        super(CIN, self).__init__(**kwargs)
        self.layer_sizes = (input_size,) + tuple(layer_sizes)
        self.direct = direct
        self.activation = activation
        self.bn = bn
        self.bias = bias

        for i, layer_size in enumerate(self.layer_sizes[:-1]):
            in_channels = input_size * layer_size
            if self.direct or i == len(self.layer_sizes) - 2:
                out_channels = self.layer_sizes[i + 1]
            else:
                out_channels = 2 * self.layer_sizes[i + 1]

            conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=self.bias)
            nn.init.xavier_uniform_(conv.weight)
            setattr(self, f'conv1d_{i+1}', conv)

            if self.bn:
                setattr(self, f'conv1d_bn_{i+1}', nn.BatchNorm1d(out_channels))

    def forward(self, x):
        batch_size = x.size(0)
        embed_size = x.size(2)

        final_result = []
        hidden_layers = []

        x0 = torch.transpose(x, 1, 2)  # [N, K, H0]
        hidden_layers.append(x0)
        x0 = x0.unsqueeze(-1)  # [N, K, H0, 1]

        for idx, layer_size in enumerate(self.layer_sizes[:-1]):
            xk = hidden_layers[-1].unsqueeze(2)  # [N, K, 1, H_(k-1)]
            out_product = torch.matmul(x0, xk)  # [N, K, H0, H_(k-1)]
            out_product = out_product.view(batch_size, embed_size,
                                           self.layer_sizes[0] * layer_size)  # [N, K, H0*H_(k-1)]
            out_product = out_product.transpose(1, 2)  # [N, H0*H_(k-1), K]
            conv = getattr(self, f'conv1d_{idx+1}')
            zk = conv(out_product)  # [N, Hk*2, K] or [N, Hk, K]

            if self.bn:
                bn = getattr(self, f'conv1d_bn_{idx+1}')
                zk = bn(zk)

            if self.activation == 'relu':
                zk = F.relu(zk)

            if self.direct:
                direct_connect = zk  # [N, Hk, K]
                next_hidden = zk.transpose(1, 2)  # [N, K, Hk]
            else:
                if idx != len(self.layer_sizes) - 2:
                    direct_connect, next_hidden = zk.split(self.layer_sizes[idx + 1], 1)
                    next_hidden = next_hidden.transpose(1, 2)  # [N, K, Hk]
                else:
                    direct_connect = zk  # [N, Hk, K]
                    next_hidden = 0

            final_result.append(direct_connect)
            hidden_layers.append(next_hidden)

        out = torch.cat(final_result, 1)  # [N, H1+H2+...+Hk, K]
        out = torch.sum(out, -1)  # [N, H1+H2+...+Hk]

        return out


class SelfAttention(nn.Module):
    def __init__(self, h_dim):
        super(SelfAttention, self).__init__()
        self.h_dim = h_dim
        self.main = nn.Sequential(
            nn.Linear(h_dim, 24),
            nn.ReLU(True),
            nn.Linear(24, 1)
        )

    def forward(self, encoder_outputs):
        b_size = encoder_outputs.size(0)
        attn_ene = self.main(encoder_outputs.contiguous().view(-1, self.h_dim))  # (b, s, h) -> (b * s, 1)
        attns = F.softmax(attn_ene.view(b_size, -1), dim=1).unsqueeze(2)  # (b * s, 1) -> (b, s, 1)
        out = (encoder_outputs * attns).sum(dim=1)  # (b, s, h) -> (b, h)
        return out

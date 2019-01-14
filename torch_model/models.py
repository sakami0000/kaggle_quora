import torch
import torch.nn as nn

from config import max_features, maxlen, embed_size
from torch_model.layers import Attention, CapsuleLayer, WeightDrop


class LstmGruAtten(nn.Module):
    def __init__(self, embedding_matrix):
        super(LstmGruAtten, self).__init__()

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


class WeightDropLstm(nn.Module):
    def __init__(self, embedding_matrix):
        super(WeightDropLstm, self).__init__()

        hidden_size = 60
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(0.1)

        self.lstm = WeightDrop(nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True),
                               ['weight_hh_l0'], dropout=0.1)
        self.gru = WeightDrop(nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True),
                              ['weight_hh_l0'], dropout=0.1)

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
        h_gru, hh_gru = self.gru(h_lstm)

        hh_gru = hh_gru.view(-1, self.hidden_size * 2)
        h_gru_atten = self.gru_attention(h_gru)

        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)

        f = torch.tensor(x[1], dtype=torch.float).cuda()

        conc = torch.cat((hh_gru, h_gru_atten, avg_pool, max_pool, f), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        conc = self.bn(conc)
        out = self.out(conc)

        return out


class CapsuleNet(nn.Module):
    def __init__(self, embedding_matrix):
        super(CapsuleNet, self).__init__()

        hidden_size = 60
        caps_out = 1

        num_capsule = 5
        dim_capsule = 10

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.1)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(hidden_size * 2, maxlen)
        self.gru_attention = Attention(hidden_size * 2, maxlen)

        self.lincaps = nn.Linear(num_capsule * dim_capsule, caps_out)
        self.caps_layer = CapsuleLayer(input_dim_capsule=hidden_size * 2,
                                       num_capsule=num_capsule,
                                       dim_capsule=dim_capsule,
                                       routings=5)

        self.linear = nn.Linear(hidden_size * 8 + caps_out + 1, 16)
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

        content3 = self.caps_layer(h_gru)
        content3 = self.dropout(content3)
        batch_size = content3.size(0)
        content3 = content3.view(batch_size, -1)
        content3 = self.relu(self.lincaps(content3))

        f = torch.tensor(x[1], dtype=torch.float).cuda()

        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool, content3, f), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        conc = self.bn(conc)
        out = self.out(conc)

        return out

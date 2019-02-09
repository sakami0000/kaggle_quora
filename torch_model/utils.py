import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class SentenceLengthDataset(Dataset):
    def __init__(self, dataset, data_len, data_features):
        self.dataset = dataset
        self.data_len = data_len
        self.data_features = data_features

    def __getitem__(self, index):
        data = self.dataset[index]
        data_len = self.data_len[index]
        data_features = self.data_features[index]
        return data, data_len, data_features

    def __len__(self):
        return len(self.dataset)

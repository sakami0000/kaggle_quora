import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogCosh(nn.Module):
    def __init__(self):
        super(LogCosh, self).__init__()

    def forward(self, preds, target):

        def _logcosh(x):
            return x + F.softplus(-2. * x) - np.log(2.)

        return torch.mean(_logcosh(target - preds))

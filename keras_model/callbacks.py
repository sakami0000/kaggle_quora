import os
import math

import numpy as np
from keras import backend as K
from keras.callbacks import Callback


class Snapshot(Callback):
    def __init__(self, folder_path, nb_epochs, nb_cycles=5, verbose=0):
        if nb_cycles > nb_epochs:
            raise ValueError('nb_epochs has to be lower than nb_cycles.')

        super(Snapshot, self).__init__()
        self.verbose = verbose
        self.folder_path = folder_path
        self.nb_epochs = nb_epochs
        self.nb_cycles = nb_cycles
        self.period = self.nb_epochs // self.nb_cycles
        self.nb_digits = len(str(self.nb_cycles))
        self.path_format = os.path.join(self.folder_path, 'weights_cycle_{}.h5')

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or (epoch + 1) % self.period != 0: return
        # only save at the end of a cycle, a not at the beginning

        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        cycle = int(epoch / self.period)
        cycle_str = str(cycle).rjust(self.nb_digits, '0')
        self.model.save_weights(self.path_format.format(cycle_str), overwrite=True)

        # resetting the learning rate
        K.set_value(self.model.optimizer.lr, self.base_lr)

        if self.verbose > 0:
            print('\nEpoch %05d: Reached %d-th cycle, saving model.' % (epoch, cycle))

    def on_epoch_begin(self, epoch, logs=None):
        if epoch <= 0: return

        lr = self.schedule(epoch)
        K.set_value(self.model.optimizer.lr, lr)

        if self.verbose > 0:
            print('\nEpoch %05d: Snapshot modifying learning '
                  'rate to %s.' % (epoch + 1, lr))

    def set_model(self, model):
        self.model = model
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        # get initial learning rate
        self.base_lr = float(K.get_value(self.model.optimizer.lr))

    def schedule(self, epoch):
        lr = math.pi * (epoch % self.period) / self.period
        lr = self.base_lr / 2 * (math.cos(lr) + 1)
        return lr


class CyclicLR(Callback):
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma

        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None, new_step_size=None):

        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size

        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
from numpy.random import RandomState

# origin
class QuatDE(Model):
    def __init__(self, config):
        super(QuatDE, self).__init__(config)

        self.ent = nn.Embedding(self.config.entTotal, self.config.hidden_size * 4)
        self.rel = nn.Embedding(self.config.relTotal, self.config.hidden_size * 4)
        self.ent_transfer = nn.Embedding(self.config.entTotal, self.config.hidden_size * 4)
        self.rel_transfer = nn.Embedding(self.config.relTotal, self.config.hidden_size * 4)
        self.rel_w = nn.Embedding(self.config.relTotal, self.config.hidden_size)

        self.criterion = nn.Softplus()
        self.fc = nn.Linear(100, 50, bias=False)
        self.ent_dropout = torch.nn.Dropout(self.config.ent_dropout)
        self.rel_dropout = torch.nn.Dropout(self.config.rel_dropout)
        self.bn = torch.nn.BatchNorm1d(self.config.hidden_size)
        self.init_weights()

    def init_weights(self):
        if True:
            r, i, j, k = self.quaternion_init(self.config.entTotal, self.config.hidden_size)
            r, i, j, k = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)
            vec1 = torch.cat([r, i, j, k], dim=1)
            self.ent.weight.data = vec1.type_as(self.ent.weight.data)
            self.ent_transfer.weight.data = vec1.type_as(self.ent_transfer.weight.data)

            s, x, y, z = self.quaternion_init(self.config.relTotal, self.config.hidden_size)
            s, x, y, z = torch.from_numpy(s), torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)
            vec2 = torch.cat([s, x, y, z], dim=1)
            self.rel.data = vec2.type_as(self.rel.weight.data)
            self.rel_transfer.data = vec2.type_as(self.rel_transfer.weight.data)
            nn.init.xavier_uniform_(self.rel_w.weight.data)
        else:
            nn.init.xavier_uniform_(self.ent.weight.data)
            nn.init.xavier_uniform_(self.rel.weight.data)
            nn.init.xavier_uniform_(self.ent_transfer.weight.data)
            nn.init.xavier_uniform_(self.rel_transfer.weight.data)

    def _calc(self, h, r):
        s_a, x_a, y_a, z_a = torch.chunk(h, 4, dim=1)
        s_b, x_b, y_b, z_b = torch.chunk(r, 4, dim=1)

        denominator_b = torch.sqrt(s_b ** 2 + x_b ** 2 + y_b ** 2 + z_b ** 2)
        s_b = s_b / denominator_b
        x_b = x_b / denominator_b
        y_b = y_b / denominator_b
        z_b = z_b / denominator_b

        A = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
        B = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
        C = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
        D = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a

        return torch.cat([A, B, C, D], dim=1)

    def loss(self, score, regul):
        # self.batch_y = ((1.0-0.1)*self.batch_y) + (1.0/self.batch_y.size(1)) /// (1 + (1 + self.batch_y)/2) * 
        return (
                torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul
        )

    def regulation(self, x):
        a, b, c, d = torch.chunk(x, 4, dim=1)
        score = torch.mean(a ** 2) + torch.mean(b ** 2) + torch.mean(c ** 2) + torch.mean(d ** 2)
        return score
    def _transfer(self, x, x_transfer, r_transfer):
        ent_transfer = self._calc(x, x_transfer)
        ent_rel_transfer = self._calc(ent_transfer, r_transfer)

        return ent_rel_transfer

    def forward(self):
        # (h, r, t) embedding
        h = self.ent(self.batch_h)
        r = self.rel(self.batch_r)
        t = self.ent(self.batch_t)

        # (h, r, t) transfer vector
        h_transfer = self.ent_transfer(self.batch_h)
        t_transfer = self.ent_transfer(self.batch_t)
        r_transfer = self.rel_transfer(self.batch_r)
        h1 = self._transfer(h, h_transfer, r_transfer)
        t1 = self._transfer(t, t_transfer, r_transfer)

        # multiplication as QuatE
        hr = self._calc(h1, r)
        # Inner product as QuatE
        score = torch.sum(hr * t1, -1)

        regul = self.regulation(h) + self.regulation(r) + self.regulation(t) + \
                self.regulation(h_transfer) + self.regulation(r_transfer) + self.regulation(t_transfer)

        return self.loss(score, regul)

    def predict(self):
        h = self.ent(self.batch_h)
        r = self.rel(self.batch_r)
        t = self.ent(self.batch_t)

        h_transfer = self.ent_transfer(self.batch_h)
        t_transfer = self.ent_transfer(self.batch_t)
        r_transfer = self.rel_transfer(self.batch_r)
        h1 = self._transfer(h, h_transfer, r_transfer)
        t1 = self._transfer(t, t_transfer, r_transfer)

        hr = self._calc(h1, r)
        score = torch.sum(hr * t1, -1)

        return score.cpu().data.numpy()

    def quaternion_init(self, in_features, out_features, criterion='he'):

        fan_in = in_features
        fan_out = out_features

        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
        rng = RandomState(123)

        # Generating randoms and purely imaginary quaternions :
        kernel_shape = (in_features, out_features)

        number_of_weights = np.prod(kernel_shape)
        v_i = np.random.uniform(0.0, 1.0, number_of_weights)
        v_j = np.random.uniform(0.0, 1.0, number_of_weights)
        v_k = np.random.uniform(0.0, 1.0, number_of_weights)

        # Purely imaginary quaternions unitary
        for i in range(0, number_of_weights):
            norm = np.sqrt(v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2) + 0.0001
            v_i[i] /= norm
            v_j[i] /= norm
            v_k[i] /= norm
        v_i = v_i.reshape(kernel_shape)
        v_j = v_j.reshape(kernel_shape)
        v_k = v_k.reshape(kernel_shape)

        modulus = rng.uniform(low=-s, high=s, size=kernel_shape)
        phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)

        weight_r = modulus * np.cos(phase)
        weight_i = modulus * v_i * np.sin(phase)
        weight_j = modulus * v_j * np.sin(phase)
        weight_k = modulus * v_k * np.sin(phase)

        return (weight_r, weight_i, weight_j, weight_k)
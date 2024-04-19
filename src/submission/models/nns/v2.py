import torch
import os
script_directory = os.path.dirname(os.path.abspath(__file__))

if 'solution' in script_directory:
    from solution import utils as ut
else:
    from submission import utils as ut
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, z_dim, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(3072 + y_dim, 500),
            nn.ELU(),
            nn.Linear(500, 500),
            nn.ELU(),
            nn.Linear(500, 2 * z_dim),
        )

    def forward(self, x, y=None):
        xy = x if y is None else torch.cat((x, y), dim=1)
        h = self.net(xy)
        m, v = ut.gaussian_parameters(h, dim=1)
        return m, v

class Decoder(nn.Module):
    def __init__(self, z_dim, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + y_dim, 500),
            nn.ELU(),
            nn.Linear(500, 500),
            nn.ELU(),
            nn.Linear(500, 3072)
        )

    def forward(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        return self.net(zy)

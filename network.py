import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def standard_normalize(x):
    x_mu = torch.mean(x, dim=1, keepdim=True)
    x_std = torch.std(x, dim=1, keepdim=True)
    return (x-x_mu)/(x_std+1e-12)

class Generator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(in_dim, 2*in_dim),
            nn.LayerNorm([2*in_dim]),
            nn.LeakyReLU(0.2),
            nn.Linear(2*in_dim, 2*in_dim),
            nn.LayerNorm([2*in_dim]),
            nn.LeakyReLU(0.2),
            nn.Linear(2*in_dim, 2*in_dim),
            nn.LayerNorm([2*in_dim]),
            nn.LeakyReLU(0.2),
            nn.Linear(2*in_dim, 2*in_dim),
            nn.LayerNorm([2*in_dim]),
            nn.LeakyReLU(0.2),
            nn.Linear(2*in_dim, out_dim),
            nn.LayerNorm([out_dim]),
            nn.LeakyReLU(0.2),
            nn.Linear(out_dim, out_dim, bias=False),
        )
        self.initialize_module(self)
    def initialize_module(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                # nn.init.uniform_(m.weight, -0.02, 0.02)
                # nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                # nn.init.normal_(m.weight)
                if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    def forward(self, x):
        out = self.module(x)
        return out

class BatchDiscriminator(nn.Module):
    def __init__(self, feature_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(feature_dim, 2*feature_dim)),
            # nn.LayerNorm([2*in_dim]),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(2*feature_dim, 2*feature_dim)),
            # nn.LayerNorm([2*in_dim]),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(2*feature_dim, 2*feature_dim)),
            # nn.LayerNorm([2*in_dim]),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(2*feature_dim, 2*feature_dim)),
            # nn.LayerNorm([2*in_dim]),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(2*feature_dim, latent_dim)),
        )

        self.single_logit = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(latent_dim, 2*latent_dim)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(2*latent_dim, latent_dim)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(latent_dim, 1, bias=False))
        )
        self.union_batch = 2
        self.union_layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(self.union_batch, self.union_batch*2, 1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv1d(self.union_batch*2, 1, 1)),
            nn.LeakyReLU(0.2),
            # nn.utils.spectral_norm(nn.Conv1d(32, 16, 1)),
            # nn.ELU(),
            # nn.utils.spectral_norm(nn.Conv1d(16, 16, 17)),
            # nn.ELU(),
            # nn.utils.spectral_norm(nn.Conv1d(16, 16, 17)),
            # nn.ELU(),
        )
        self.union_logit = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(self.union_batch, self.union_batch*2, 1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv1d(self.union_batch*2, 1, 1)),
            nn.LeakyReLU(0.2),
            nn.Flatten(),

            nn.utils.spectral_norm(nn.Linear(latent_dim, 2*latent_dim)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(2*latent_dim, latent_dim)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(latent_dim, 1, bias=False))
        )
        # self.initialize_module(self)
    def initialize_module(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                # nn.init.uniform_(m.weight, -0.02, 0.02)
                # nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                # nn.init.normal_(m.weight)
                if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    def forward(self, x):
        out = self.encoder(x)
        single_logit = self.single_logit(out)

        out = out.view(out.shape[0]//self.union_batch, self.union_batch, -1)
        union_logit = self.union_logit(out)
        union_logit = union_logit.repeat(self.union_batch, 1)

        logit = torch.concat([union_logit, single_logit], dim=1)
        return logit
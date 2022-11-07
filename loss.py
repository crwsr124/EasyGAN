import torch
import math

device = 'cuda'

smooth_l1_loss = torch.nn.SmoothL1Loss().to(device)
l1_loss = torch.nn.L1Loss().to(device)
mse_loss = torch.nn.MSELoss().to(device)
bce_logits_loss = torch.nn.BCEWithLogitsLoss().to(device)


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)

    def forward(self, input, target):
        pt = torch.sigmoid(input)
        logp = self.ce(input, target)
        loss = (1 - pt) ** self.gamma * logp
        return loss.mean()
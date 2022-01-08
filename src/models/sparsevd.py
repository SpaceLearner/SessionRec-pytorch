import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class LinearSVDO(nn.Module):
    def __init__(self, in_features, out_features, threshold, bias=True, name=None):
        super(LinearSVDO, self).__init__()
        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        
        self.hasbias = bias

        self.W = Parameter(torch.Tensor(out_features, in_features))
        self.log_sigma = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(1, out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        self.bias.data.zero_()
        self.W.data.normal_(0, 0.02)
        # nn.init.xavier_uniform_(self.W)
        self.log_sigma.data.fill_(-5)        
        
    def forward(self, x):
        self.log_alpha = self.log_sigma * 2.0 - 2.0 * torch.log(1e-16 + torch.abs(self.W))
        self.log_alpha = torch.clamp(self.log_alpha, -10, 10) 
        
        if self.training:
            lrt_mean =  F.linear(x, self.W) + self.bias
            lrt_std = torch.sqrt(F.linear(x * x, torch.exp(self.log_sigma * 2.0)) + 1e-8)
            eps = lrt_std.data.new(lrt_std.size()).normal_()
            return lrt_mean + lrt_std * eps
    
        return F.linear(x, self.W * (self.log_alpha < 8.0).float()) + self.bias if self.hasbias else F.linear(x, self.W * (self.log_alpha < 8.0).float()) 
        # return F.linear(x, self.W) + self.bias if self.hasbias else F.linear(x, self.W)
        
    def kl_reg(self):
        # Return KL here -- a scalar 
        k1, k2, k3 = torch.Tensor([0.63576]).to(self.W.device), torch.Tensor([1.8732]).to(self.W.device), torch.Tensor([1.48695]).to(self.W.device)
        kl = k1 * torch.sigmoid(k2 + k3 * self.log_alpha) - 0.5 * torch.log1p(torch.exp(-self.log_alpha))
        a = - torch.sum(kl)
        return a
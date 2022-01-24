import torch.nn as nn
import torch.nn.functional as F
from .sparsevd import LinearSVDO
import math

class GRUCell(nn.Module):
    
    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, bias=True, threshold=8.0, name=None):
        super(GRUCell, self).__init__()
        self.name = name
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = LinearSVDO(input_size,  3 * hidden_size, bias=bias, threshold=threshold, name=name+"_x2h")
        self.h2h = LinearSVDO(hidden_size, 3 * hidden_size, bias=bias, threshold=threshold, name=name+"_h2h")
        self.reset_parameters()



    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        
        x = x.view(-1, x.size(1))
        
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hidden)
        
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        
        
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))
        
        hy = newgate + inputgate * (hidden - newgate)
        
        
        return hy
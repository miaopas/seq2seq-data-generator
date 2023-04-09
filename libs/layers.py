import torch.nn as nn
import torch
from torch.nn.modules import RNN
import math


class Module(nn.Module):
    '''
        Wraper to extend functions of models.
    '''
    def __init__(self):
        super().__init__()

    def count_parameters(self):
        param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'The model has {param:,} trainable parameters')


class LinearRNN(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hid_dim,
                ):

        super().__init__()
        
        self.input_ff = nn.Linear(input_dim, hid_dim, bias=False)
        self.hidden_ff = nn.Linear(hid_dim,hid_dim, bias=False)
        self.output_ff = nn.Linear(hid_dim, output_dim, bias=False)

        self.hidden_ff.weight.data = torch.diag(torch.rand(hid_dim))

        self.hid_dim = hid_dim

    def forward(self, x):
        
        #src = [batch size, input len, input dim]
        length = x.shape[1]

        hidden = []
        hidden.append(torch.zeros(1, 1, self.hid_dim, dtype=x.dtype, device=x.device))
        
        x = self.input_ff(x)

        for i in range(length):
            h_next = x[:,i:i+1,:] + self.hidden_ff(hidden[i])
            hidden.append(h_next)

        hidden = torch.cat(hidden[1:], dim=1)
        out = self.output_ff(hidden)
        return out
    
class ComplexLinearRNN(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hid_dim,
                ):

        super().__init__()
        
        self.input_ff = nn.Linear(input_dim, hid_dim, bias=False,dtype=torch.cdouble)
        self.hidden_ff = nn.Linear(hid_dim,hid_dim, bias=False,dtype=torch.cdouble)
        self.output_ff = nn.Linear(hid_dim, output_dim, bias=False,dtype=torch.cdouble)

        # self.hidden_ff.weight.data = torch.diag(torch.rand(hid_dim,dtype=torch.cdouble))

        self.hid_dim = hid_dim

    def forward(self, x):
        
        #src = [batch size, input len, input dim]
        length = x.shape[1]

        hidden = []
        hidden.append(torch.zeros(1, 1, self.hid_dim, dtype=x.dtype, device=x.device))
        
        x = self.input_ff(x)

        for i in range(length):
            h_next = x[:,i:i+1,:] + 1.0j * self.hidden_ff(hidden[i])
            hidden.append(h_next)

        hidden = torch.cat(hidden[1:], dim=1)
        out = self.output_ff(hidden)
        return out
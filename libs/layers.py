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
    
class LinearTCN(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 channel,
                 layers
                ):

        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.input_conv = nn.Conv1d(input_dim, channel, 2, stride=1, padding=1, dilation=1, groups=input_dim, bias=False)
        self.conv_filters = nn.ModuleList([nn.Conv1d(channel, channel, 2, stride=1, padding=2**i, dilation=2**i, groups=channel, bias=False) for i in range(1, layers)])
        
        self.out = nn.Linear(channel, output_dim, bias=False)

    def forward(self, x):
        
        # Conv1d is (N, C, L)

        x = x.permute(0,2,1).contiguous()


        x = self.input_conv(x)
        x = x[:, :, :-2**0].contiguous()
        for i, filter in enumerate(self.conv_filters, start=1):
            x = filter(x)
            x = x[:, :, :-2**i].contiguous()
        
        x = x.permute(0,2,1).contiguous()
        out = self.out(x)

        return out
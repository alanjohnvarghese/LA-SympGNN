import numpy as np
import math
import torch
from torch.nn import Linear, Parameter, Module, init
import torch.nn.functional as F 
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class SymmetricLinear(Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ##### Change: Assert that in_features == out_features #####
        assert in_features==out_features, f"Expects in_features = out_features, got {in_features} and {out_features}"
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        ##### Change: Wnew = W + W^T #####
        return F.linear(input, self.weight + self.weight.t(), self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class SymplecticMessagePassing(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr = 'add')
        self.lin = SymmetricLinear(in_channels, out_channels, bias = False)
        self.bias= Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):

        #x: [N, in_channels]
        #edge_index: [2,E]

        #edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))

        x = self.lin(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, norm=norm)

        out = out + self.bias

        return out

    def message(self, x_j, norm):
        #x_j: [E, out_channels]
        return norm.view(-1,1)*x_j


class LAMessagePassingUp(torch.nn.Module):
    def __init__(self,n):
        super(LAMessagePassingUp, self).__init__()
        self.message_passing = SymplecticMessagePassing(n,n)
        

    def forward(self,p,q,edge_index):
        p = p + self.message_passing(q, edge_index)
        q = q
        return p,q


class LAMessagePassingDown(torch.nn.Module):
    def __init__(self,n):
        super(LAMessagePassingDown, self).__init__()
        self.message_passing = SymplecticMessagePassing(n,n)

    def forward(self,p,q,edge_index):
        p = p
        q = q + self.message_passing(p, edge_index)
        return p,q

class LAMessagePassingDown_NoInteraction(torch.nn.Module):
    def __init__(self,n):
        super(LAMessagePassingDown_NoInteraction, self).__init__()
        self.linear = SymmetricLinear(n,n, bias = False)

    def forward(self,p,q,edge_index):
        p = p
        q = q + self.linear(p)
        return p,q
        

class ActivationUpModule(torch.nn.Module):
    def __init__(self,n):
        super(ActivationUpModule,self).__init__()
        self.a = torch.nn.Parameter(torch.ones((1,n), requires_grad = True))
        self.activation = torch.nn.Tanh()

    def forward(self,p,q):
        sigma_q = self.activation(q)
        p = p + self.a*sigma_q
        q = q
        return p,q
        
class ActivationDownModule(torch.nn.Module):
    def __init__(self,n):
        super(ActivationDownModule,self).__init__()
        self.a = torch.nn.Parameter(torch.ones((1,n), requires_grad = True))
        self.activation = torch.nn.Tanh()

    def forward(self,p,q):
        sigma_p = self.activation(p)
        p = p
        q = self.a*sigma_p + q
        return p,q

class LACombinedUp(torch.nn.Module):
    def __init__(self,n):
        super(LACombinedUp, self).__init__()
        self.message_passing = SymplecticMessagePassing(n,n)
        self.a = torch.nn.Parameter(torch.ones((1,n), requires_grad = True))
        self.activation = torch.nn.Tanh()

    def forward(self, p, q, edge_index):
        q_prime = self.message_passing(q, edge_index)
        q_prime = q_prime + self.a*q
        p = p + self.activation(q_prime)
        q = q
        return p,q

class LACombinedDown(torch.nn.Module):
    def __init__(self,n):
        super(LACombinedDown, self).__init__()
        self.message_passing = SymplecticMessagePassing(n,n)
        self.a = torch.nn.Parameter(torch.ones((1,n), requires_grad = True))
        self.activation = torch.nn.Tanh()

    def forward(self, p, q, edge_index):
        p_prime = self.message_passing(p, edge_index)   #[n,d]
        p_prime = p_prime + self.a*p                    #[n,d] + [1,d]*[n,d]
        p = p
        q = q + self.activation(p_prime)
        return p,q

############# Attention##########

class SymplecticMessagePassingAttention(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr = 'add')
        self.lin = SymmetricLinear(in_channels, out_channels, bias = False)
        self.bias= Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, att_scores):

        #x: [N, in_channels]
        #edge_index: [2,E]

        #edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))

        x = self.lin(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]*att_scores
        #print(norm)
        out = self.propagate(edge_index, x=x, norm=norm)

        out = out + self.bias

        return out

    def message(self, x_j, norm):
        #x_j: [E, out_channels]
        return norm.view(-1,1)*x_j

class LAMessagePassingUpAttention(torch.nn.Module):
    def __init__(self,n):
        super(LAMessagePassingUpAttention, self).__init__()
        self.message_passing = SymplecticMessagePassingAttention(n,n)
        

    def forward(self,p,q,edge_index, att_scores):
        p = p + self.message_passing(q, edge_index, att_scores)
        q = q
        return p,q

class LAMessagePassingDownAttention(torch.nn.Module):
    def __init__(self,n):
        super(LAMessagePassingDownAttention, self).__init__()
        self.message_passing = SymplecticMessagePassingAttention(n,n)

    def forward(self,p,q,edge_index, att_scores):
        p = p
        q = q + self.message_passing(p, edge_index, att_scores)
        return p,q

class Attention(torch.nn.Module):
    def __init__(self,n):
        super(Attention, self).__init__()
        self.QK = torch.nn.Linear(n,n, bias = False)

    def forward(self, x, edge_index):
        Q = self.QK(x)
        K = self.QK(x)
        attention_scores = torch.sum(Q[edge_index[0]] * K[edge_index[1]], dim = 1)
        return attention_scores


# class DropoutUpModule(torch.nn.Module):
#     def __init__self(,n

class LACombinedUpSource(torch.nn.Module):
    def __init__(self,n):
        super(LACombinedUpSource, self).__init__()
        self.message_passing = SymplecticMessagePassing(n,n)
        self.a = torch.nn.Parameter(torch.ones((1,n), requires_grad = True))
        self.activation = torch.nn.Tanh()

    def forward(self, p, q, edge_index, source):
        q_prime = self.message_passing(q, edge_index)
        q_prime = q_prime + self.a*q + source
        p = p + self.activation(q_prime)
        q = q
        return p,q

class LACombinedDownSource(torch.nn.Module):
    def __init__(self,n):
        super(LACombinedDownSource, self).__init__()
        self.message_passing = SymplecticMessagePassing(n,n)
        self.a = torch.nn.Parameter(torch.ones((1,n), requires_grad = True))
        self.activation = torch.nn.Tanh()

    def forward(self, p, q, edge_index, source):
        p_prime = self.message_passing(p, edge_index)   #[n,d]
        p_prime = p_prime + self.a*p + source                   #[n,d] + [1,d]*[n,d]
        p = p
        q = q + self.activation(p_prime)
        return p,q
        
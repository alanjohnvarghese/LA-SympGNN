import numpy as np
import math
import torch
from torch.nn import Linear, Parameter, Module, init
import torch.nn.functional as F 
from torch import Tensor
from modules import *


class SGNN(torch.nn.Module):
    def __init__(self,opt):
        super(SGNN,self).__init__()

        self.hidden  = opt["hidden"]
        self.num_features= opt["num_features"]
        self.num_classes = opt["num_classes"]

        self.dropout1 = torch.nn.Dropout(opt["input_dropout"])
        self.dropout2 = torch.nn.Dropout(opt["input_dropout"])

        self.linear_p = torch.nn.Linear(self.num_features,self.hidden, bias = False)
        self.linear_q = torch.nn.Linear(self.num_features,self.hidden, bias = False)
        
        self.layers = torch.nn.ModuleList()
        for _ in range(opt["num_h_layers"]):
            self.layers.append(torch.nn.ModuleList([
                LAMessagePassingUp(self.hidden),
                LAMessagePassingDown(self.hidden),
                ActivationUpModule(self.hidden),
                ActivationDownModule(self.hidden)
            ]))

        self.layers.append(torch.nn.ModuleList([
            LAMessagePassingUp(self.hidden),
            LAMessagePassingDown(self.hidden)
            ]))
        
        self.linear = torch.nn.Linear(self.hidden,self.num_classes)

    def forward(self, x, edge_index):
        p = self.linear_p(x)
        q = self.linear_q(x)
        
        p = self.dropout1(p)
        q = self.dropout2(q)
        
        for layer in self.layers[:-1]:
            la_up, la_down, actvn_up, actvn_down = layer
            p,q = la_up(p,q,edge_index)
            p,q = la_down(p,q,edge_index)
            p,q = actvn_up(p,q)
            p,q = actvn_down(p,q)
        
        la_up,la_down = self.layers[-1]
        p,q = la_up(p,q,edge_index)
        p,q = la_down(p,q,edge_index)
        

        out = self.linear(q)

        return out

class SGNN_NoInteraction(torch.nn.Module):
    def __init__(self,opt):
        super(SGNN_NoInteraction,self).__init__()

        self.hidden  = opt["hidden"]
        self.num_features= opt["num_features"]
        self.num_classes = opt["num_classes"]

        self.dropout1 = torch.nn.Dropout(opt["input_dropout"])
        self.dropout2 = torch.nn.Dropout(opt["input_dropout"])

        self.linear_p = torch.nn.Linear(self.num_features,self.hidden, bias = False)
        self.linear_q = torch.nn.Linear(self.num_features,self.hidden, bias = False)
        
        self.layers = torch.nn.ModuleList()
        for _ in range(opt["num_h_layers"]):
            self.layers.append(torch.nn.ModuleList([
                LAMessagePassingUp(self.hidden),
                LAMessagePassingDown_NoInteraction(self.hidden),
                ActivationUpModule(self.hidden),
                ActivationDownModule(self.hidden)
            ]))

        self.layers.append(torch.nn.ModuleList([
            LAMessagePassingUp(self.hidden),
            LAMessagePassingDown_NoInteraction(self.hidden)
            ]))
        
        self.linear = torch.nn.Linear(self.hidden,self.num_classes)

    def forward(self, x, edge_index):
        p = self.linear_p(x)
        q = self.linear_q(x)
        
        p = self.dropout1(p)
        q = self.dropout2(q)
        
        for layer in self.layers[:-1]:
            la_up, la_down, actvn_up, actvn_down = layer
            p,q = la_up(p,q,edge_index)
            p,q = la_down(p,q,edge_index)
            p,q = actvn_up(p,q)
            p,q = actvn_down(p,q)
        
        la_up,la_down = self.layers[-1]
        p,q = la_up(p,q,edge_index)
        p,q = la_down(p,q,edge_index)
        

        out = self.linear(q)

        return out

class SGNN_version2(torch.nn.Module):
    def __init__(self,opt):
        super(SGNN_version2,self).__init__()

        self.hidden  = opt["hidden"]
        self.num_features= opt["num_features"]
        self.num_classes = opt["num_classes"]

        self.dropout1 = torch.nn.Dropout(opt["input_dropout"])
        self.dropout2 = torch.nn.Dropout(opt["input_dropout"])

        self.linear_p = torch.nn.Linear(self.num_features,self.hidden, bias = False)
        self.linear_q = torch.nn.Linear(self.num_features,self.hidden, bias = False)
        
        self.layers = torch.nn.ModuleList()
        for _ in range(opt["num_h_layers"]):
            self.layers.append(torch.nn.ModuleList([
                LAMessagePassingUp(self.hidden),
                ActivationUpModule(self.hidden),
                LAMessagePassingDown(self.hidden),
                ActivationDownModule(self.hidden)
            ]))

        # self.layers.append(torch.nn.ModuleList([
        #     LAMessagePassingUp(self.hidden),
        #     LAMessagePassingDown(self.hidden)
        #     ]))
        
        self.linear = torch.nn.Linear(self.hidden,self.num_classes)

    def forward(self, x, edge_index):
        p = self.linear_p(x)
        q = self.linear_q(x)
        
        p = self.dropout1(p)
        q = self.dropout2(q)
        
        for layer in self.layers:
            la_up, actvn_up, la_down, actvn_down = layer
            p,q = la_up(p,q,edge_index)
            p,q = actvn_up(p,q)
            p,q = la_down(p,q,edge_index)
            p,q = actvn_down(p,q)
        
        # la_up,la_down = self.layers[-1]
        # p,q = la_up(p,q,edge_index)
        # p,q = la_down(p,q,edge_index)
        

        out = self.linear(q)

        return out


class SGNN_version3(torch.nn.Module):
    def __init__(self,opt):
        super(SGNN_version3, self).__init__()

        self.hidden  = opt["hidden"]
        self.num_features= opt["num_features"]
        self.num_classes = opt["num_classes"]

        self.dropout1 = torch.nn.Dropout(opt["input_dropout"])
        self.dropout2 = torch.nn.Dropout(opt["input_dropout"])

        self.linear_p = torch.nn.Linear(self.num_features,self.hidden, bias = False)
        self.linear_q = torch.nn.Linear(self.num_features,self.hidden, bias = False)
        
        self.layers = torch.nn.ModuleList()
        for _ in range(opt["num_h_layers"]):
            self.layers.append(torch.nn.ModuleList([
                LACombinedUp(self.hidden),
                LACombinedDown(self.hidden)
            ]))

        # self.layers.append(torch.nn.ModuleList([
        #     LAMessagePassingUp(self.hidden),
        #     LAMessagePassingDown(self.hidden)
        #     ]))
        
        self.linear = torch.nn.Linear(self.hidden,self.num_classes)

    def forward(self, x, edge_index):
        p = self.linear_p(x)
        q = self.linear_q(x)
        
        p = self.dropout1(p)
        q = self.dropout2(q)
        
        for layer in self.layers:
            la_up, la_down = layer
            p,q = la_up(p,q,edge_index)
            p,q = la_down(p,q,edge_index)
        
        # la_up,la_down = self.layers[-1]
        # p,q = la_up(p,q,edge_index)
        # p,q = la_down(p,q,edge_index)
        

        out = self.linear(q)

        return out


class SGNN_Attention(torch.nn.Module):
    def __init__(self,opt):
        super(SGNN_Attention,self).__init__()

        self.hidden  = opt["hidden"]
        self.num_features= opt["num_features"]
        self.num_classes = opt["num_classes"]

        self.attention = Attention(self.num_features)

        self.dropout1 = torch.nn.Dropout(opt["input_dropout"])
        self.dropout2 = torch.nn.Dropout(opt["input_dropout"])

        self.linear_p = torch.nn.Linear(self.num_features,self.hidden, bias = False)
        self.linear_q = torch.nn.Linear(self.num_features,self.hidden, bias = False)
        
        self.layers = torch.nn.ModuleList()
        for _ in range(opt["num_h_layers"]):
            self.layers.append(torch.nn.ModuleList([
                LAMessagePassingUpAttention(self.hidden),
                LAMessagePassingDownAttention(self.hidden),
                ActivationUpModule(self.hidden),
                ActivationDownModule(self.hidden)
            ]))

        self.layers.append(torch.nn.ModuleList([
            LAMessagePassingUpAttention(self.hidden),
            LAMessagePassingDownAttention(self.hidden)
            ]))
        
        self.linear = torch.nn.Linear(self.hidden,self.num_classes)

    def forward(self, x, edge_index):
        att_scores = self.attention(x,edge_index)
        p = self.linear_p(x)
        q = self.linear_q(x)
        
        p = self.dropout1(p)
        q = self.dropout2(q)
        
        for layer in self.layers[:-1]:
            la_up, la_down, actvn_up, actvn_down = layer
            p,q = la_up(p,q,edge_index, att_scores)
            p,q = la_down(p,q,edge_index, att_scores)
            p,q = actvn_up(p,q)
            p,q = actvn_down(p,q)
        
        la_up,la_down = self.layers[-1]
        p,q = la_up(p,q,edge_index, att_scores)
        p,q = la_down(p,q,edge_index, att_scores)
        

        out = self.linear(q)

        return out


class SGNN_source(torch.nn.Module):
    def __init__(self,opt):
        super(SGNN_source, self).__init__()

        self.hidden  = opt["hidden"]
        self.num_features= opt["num_features"]
        self.num_classes = opt["num_classes"]

        self.dropout1 = torch.nn.Dropout(opt["input_dropout"])
        self.dropout2 = torch.nn.Dropout(opt["input_dropout"])

        self.beta = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.linear_p = torch.nn.Linear(self.num_features,self.hidden, bias = False)
        self.linear_q = torch.nn.Linear(self.num_features,self.hidden, bias = False)
        
        self.layers = torch.nn.ModuleList()
        for _ in range(opt["num_h_layers"]):
            self.layers.append(torch.nn.ModuleList([
                LACombinedUpSource(self.hidden),
                LACombinedDownSource(self.hidden)
            ]))

        # self.layers.append(torch.nn.ModuleList([
        #     LAMessagePassingUp(self.hidden),
        #     LAMessagePassingDown(self.hidden)
        #     ]))
        
        self.linear = torch.nn.Linear(self.hidden,self.num_classes)

    def forward(self, x, edge_index):

        p = self.linear_p(x)
        q = self.linear_q(x)
        
        p = self.dropout1(p)
        q = self.dropout2(q)

        source = self.beta * q 
        
        for layer in self.layers:
            la_up, la_down = layer
            p,q = la_up(p,q,edge_index, source)
            p,q = la_down(p,q,edge_index, source)
        
        # la_up,la_down = self.layers[-1]
        # p,q = la_up(p,q,edge_index)
        # p,q = la_down(p,q,edge_index)
        

        out = self.linear(q)

        return out
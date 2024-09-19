import math
import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from .LoggerFactory import get_logger

logger = get_logger()

class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.weight.data, gain=1.4)
        self.w = Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.w.data, gain=1.1)
        self.a = Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.leakyrelu = torch.nn.LeakyReLU(0.2)
        self.adj_att = Parameter(torch.zeros(size=(in_features, out_features)))
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        Wh = torch.mm(x, self.w)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        adj = adj.to_dense()
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, 0.2, training=self.training)
        shape = attention.shape
        matrix = torch.ones(shape[0], shape[1]).cuda()
        attention = attention + matrix
        new_adj = adj * attention
        row_sums = new_adj.sum(dim=1)
        normalized_tensor = new_adj / row_sums.unsqueeze(1)
        support = torch.mm(normalized_tensor, x)
        output = torch.mm(support, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
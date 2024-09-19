import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution
from .LoggerFactory import get_logger

logger = get_logger()


class AoSEGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, num_layers):
        super(AoSEGCN, self).__init__()
        self.aose_gcns = nn.ModuleList([GraphConvolution(nfeat, nhid)])
        for _ in range(num_layers - 1):
            self.wagcns.append(GraphConvolution(nhid, nhid))
        self.aose_final = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        for wagcn in self.aose_gcns:
            x = F.relu(wagcn(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.aose_final(x, adj)
        return F.log_softmax(x, dim=1)

    def l2_loss(self):
        loss = None
        for layer in self.aose_gcns:
            for name, param in layer.named_parameters():
                if loss is None:
                    loss = param.pow(2).sum()
                else:
                    loss += param.pow(2).sum()
        return loss
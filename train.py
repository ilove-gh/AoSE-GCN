from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from src import cudaUtils
from src.models import AoSEGCN
from src.dataUtils import load_data, accuracy
from src.LoggerFactory import get_logger
import uuid
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

logger = get_logger()

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=66, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')

parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
                    help='Dataset string')
parser.add_argument('--path', type=str, default='./data',
                    help='Dataset string')

args = parser.parse_args()
logger.info('The list of model initialization parameters is {}'.format(args.__dict__))


checkpt_file = 'pretrained/' + uuid.uuid4().hex + '.pt'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# 判定是否有可用的GPU设备
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda")
    cudaUtils.all_cuda_infos()
    cudaUtils.current_cuda_info()
else:
    device = torch.device("cpu")

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(path=args.path,
                                                                dataset_str=args.dataset)
adj = adj.to(device)
features = features.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)

# Model and optimizer
model = AoSEGCN(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=int(labels.max()) + 1,
              dropout=args.dropout,
              num_layers=1)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train():
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train += args.weight_decay * model.l2_loss()
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), acc_train.item()

def validation():
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        return loss_val.item(), acc_val.item()


def test():
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    return acc_test.item(), loss_test.item(), output.detach().cpu().numpy()



# Train model
t_total = time.time()
bad_counter = 0
loss_best = 999999
acc_best = 0
for epoch in range(args.epochs):
    loss_train, acc_train = train()
    loss_val, acc_val = validation()
    logger.info('Epoch: {:04d}, loss_train: {:.4f}, acc_train: {:.4f}, loss_val: {:.4f}, acc_val: {:.4f}\
            '.format(epoch + 1, loss_train, acc_train, loss_val, acc_val))

    if acc_val > acc_best:
        acc_best = acc_val

        torch.save(model.state_dict(), checkpt_file)
        acc_test, loss_test, test_out = test()
        bad_counter = 0
        logger.warning("lossval results: \
                            epoch={},loss= {:.4f}, accuracy= {:.4f},test_loss={:.4f},test_acc={:.4f}".format(epoch + 1,
                                                                                                             loss_val,
                                                                                                             acc_val,
                                                                                                             loss_test,
                                                                                                             acc_test))
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

# 创建文件夹来保存模型
if not os.path.exists('./saved_models'):
    os.makedirs('./saved_models')

acc_test, loss_test, test_out = test()
torch.save(model, 'saved_models/aw-gcn_model_{}_{:.4f}.pth'.format(args.dataset, acc_test))
logger.info('Test Accuracy {:.4f}.'.format(acc_test))


logger.info("The model is optimized in the {} dataset!".format(args.dataset))
logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))


tsne = TSNE()
out = tsne.fit_transform(test_out[idx_test.cpu()])
fig = plt.figure()
for i in range(7):
    indices = labels[idx_test].cpu().detach().numpy() == i
    x, y = out[indices].T
    plt.scatter(x, y, label=str(i))
plt.legend()
plt.show()

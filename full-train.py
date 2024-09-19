from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import random
import uuid

import numpy as np
import torch
import wandb
import torch.nn.functional as F
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score

from src.process import full_load_data
from src import cudaUtils
from src.models import AoSEGCN
from src.dataUtils import accuracy
from src.LoggerFactory import get_logger

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
parser.add_argument('--weight_decay', type=float, default=7e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='texas',
                    help='Dataset string')
parser.add_argument('--patience', type=int, default=30, help='Patience')

parser.add_argument('--path', type=str, default='./data',
                    help='Dataset string')

args = parser.parse_args()
logger.info('The list of model initialization parameters is {}'.format(args.__dict__))

checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda")
    cudaUtils.all_cuda_infos()
    cudaUtils.current_cuda_info()
else:
    device = torch.device("cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# Create a folder to hold your models
if not os.path.exists('./saved_models'):
    os.makedirs('./saved_models')


def train_main(datastr, splitstr):
    # Load data
    adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = full_load_data(datastr, splitstr)

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
                  dropout=args.dropout
                  ,num_layers=1)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)

        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train += args.weight_decay * model.l2_loss()
        acc_train = accuracy(output[idx_train], labels[idx_train])

        loss_train.backward()
        optimizer.step()
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        loss_val += args.weight_decay * model.l2_loss()
        acc_val = accuracy(output[idx_val], labels[idx_val])

        logger.info('Epoch: {:04d}, loss_train: {:.4f}, acc_train: {:.4f}, loss_val: {:.4f}, acc_val: {:.4f}, time: {:.4f}s\
        '.format(epoch + 1, loss_train.item(), acc_train.item(), loss_val.item(), acc_val.item(), time.time() - t))

        return loss_train.item(), acc_train.item(), loss_val.item(), acc_val.item()

    def test():
        model.load_state_dict(torch.load(checkpt_file))
        model.eval()
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])

        logger.warning("Test set results: \
            loss= {:.4f}, accuracy= {:.4f}".format(loss_test.item(), acc_test.item()))

        return acc_test.item(), loss_test.item()

    def _test(model):
        model.load_state_dict(torch.load(checkpt_file))
        model.eval()
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])

        logger.warning("Test set results: \
            loss= {:.4f}, accuracy= {:.4f}".format(loss_test.item(), acc_test.item()))

        return acc_test.item(), loss_test.item()

    def metrics(output, labels):
        preds = output.max(1)[1].type_as(labels)
        precision = precision_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='weighted')
        recall = recall_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='weighted')
        f1 = f1_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='weighted')
        accuracy = accuracy_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy())

        return [precision, recall, f1,accuracy]


    def _score_test(model):
        model.load_state_dict(torch.load(checkpt_file))
        model.eval()
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        metrics_score = metrics(output[idx_test], labels[idx_test])

        logger.warning("_score_test set results: \
                    loss= {:.4f}, accuracy= {:.4f}".format(loss_test.item(), acc_test.item()))

        return acc_test.item(), loss_test.item(),metrics_score

    acc_best = 0
    bad_counter = 0
    for epoch in range(args.epochs):
        loss_train, acc_train, loss_val, acc_val = train(epoch)
        if acc_val > acc_best:
            acc_best = acc_val
            bad_counter = 0
            torch.save(model.state_dict(), checkpt_file)
            logger.warning("lossval results: \
                                        epoch={},loss= {:.4f}, accuracy= {:.4f}".format(
                epoch + 1, loss_val, acc_val))
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

    acc_test, loss_test = test()
    torch.save(model.load_state_dict(torch.load(checkpt_file)), 'saved_models/wagcn_model_{}_{:.4f}.pth'.format(args.dataset + splitstr.replace('/','_'), acc_test))

    return _score_test(model)


# Train model
t_total = time.time()
test_acc_list = []
precision_list = []
recall_list = []
f1_list = []
acc_list = []
for i in range(10):
    datastr = args.dataset
    print('datastr',datastr)

    splitstr = 'splits/' + args.dataset + '_split_0.6_0.2_' + str(i) + '.npz'
    print('splitstr', splitstr)
    acc_test, loss_test ,metrics_score= train_main(datastr, splitstr)
    test_acc_list.append(acc_test)
    precision_list.append(metrics_score[0])
    recall_list.append(metrics_score[1])
    f1_list.append(metrics_score[2])
    acc_list.append(metrics_score[3])

logger.success("All test acc list is :{}".format(test_acc_list))
logger.success("Test acc.:{:.4f}".format(np.mean(test_acc_list)))
logger.success("Test precision.:{:.4f}".format(np.mean(precision_list)))
logger.success("Test recall.:{:.4f}".format(np.mean(recall_list)))
logger.success("Test f1.:{:.4f}".format(np.mean(f1_list)))
logger.success("Test acc_.:{:.4f}".format(np.mean(acc_list)))

logger.info("The model is optimized in the {} dataset!".format(args.dataset))
logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
wandb.finish()
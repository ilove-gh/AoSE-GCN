import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os
import sys
import torch

from .LoggerFactory import get_logger
from sklearn.metrics import precision_score,recall_score,f1_score

logger = get_logger()


# 读取文件内所有行数据（index）转化为列表
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def rw_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).tocoo()

def add_identity_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    return adj.tocoo()


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_data(path, dataset_str):
    """
    ind.*.x : 训练集节点特征向量，保存对象为：scipy.sparse.csr.csr_matrix，实际展开后大小为： (140, 1433)<br>
    ind.*.tx : 测试集节点特征向量，保存对象为：scipy.sparse.csr.csr_matrix，实际展开后大小为： (1000, 1433)<br>
    ind.*.allx : 包含有标签和无标签的训练节点特征向量，保存对象为：scipy.sparse.csr.csr_matrix，实际展开后大小为：(1708, 1433)，可以理解为除测试集以外的其他节点特征集合，训练集是它的子集<br>
    ind.*.y : one-hot表示的训练节点的标签，保存对象为：numpy.ndarray<br>
    ind.*.ty : one-hot表示的测试节点的标签，保存对象为：numpy.ndarray<br>
    ind.*.ally : one-hot表示的ind.cora.allx对应的标签，保存对象为：numpy.ndarray<br>
    ind.*.graph : 保存节点之间边的信息，保存格式为：{ index : [ index_of_neighbor_nodes ] }<br>
    ind.*.test.index : 保存测试集节点的索引，保存对象为：List，用于后面的归纳学习设置。<br>
    """
    logger.info('Loading dataset {} from path {}...'.format(dataset_str, path))

    # step 1: 读取ind.cora.*文件后缀为 x, y, tx, ty, allx, ally, graph的文件，并保存为对象。
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        # with open语法对文件流控制做了管理，不需要手动调用close方法关闭。
        with open(os.path.join(path, "{}/ind.{}.{}".format(dataset_str, dataset_str, names[i])), 'rb') as f:
            # python 版本必须高于3.0
            if sys.version_info > (3, 0):
                # ind*.x,ind*.y,ind*.tx,ind*.ty,ind*.allx,ind*.ally,ind*.graph,类型文件为pkl文件。
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    #  ind*.x,ind*.y,ind*.tx,ind*.ty,ind*.allx,ind*.ally,ind*.graph文件对象，分别保存在x, y, tx, ty, allx, ally, graph中
    x, y, tx, ty, allx, ally, graph = tuple(objects)

    # step 2: 读取测试集索引.
    # test_idx_reorder为list列表
    test_idx_reorder = parse_index_file(os.path.join(path, "{}/ind.{}.test.index".format(dataset_str, dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    # citeseer数据集中一些孤立点的特殊处理
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    # 获取整个图的所有节点特征
    # sp.vstack按行垂直链接两个数组，tolil()为转化为lil_matrix类型矩阵
    # allx除去测试机之外的所有节点特征向量；tx测试机所有的特征向量，拼接后得到的就是全集的特征向量
    features = sp.vstack((allx, tx)).tolil()
    # 交换变量位置
    features[test_idx_reorder, :] = features[test_idx_range, :]
    # 根据自己需要归一化特征，返回list类型，[coordinates, data, shape]
    features = normalize(features)
    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()

    # 获取整个图的邻接矩阵
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # 根据自己需要归一化邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = add_identity_adjacency(adj)

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    # 获取所有节点标签,标签编码使用one-hot
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    idx_test = test_idx_range.tolist()

    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    logger.info('Loading the dataset {} from path {} is complete'.format(dataset_str, path))

    return adj, features, labels, idx_train, idx_val, idx_test


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    # print('precision_score:',
    #       precision_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='weighted'))
    # print('precision_score:',
    #       recall_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='weighted'))
    # print('precision_score:', f1_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='weighted'))

    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


if __name__ == '__main__':
    load_data(path='../data/cora', dataset_str='cora')

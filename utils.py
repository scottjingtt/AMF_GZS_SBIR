#-*- coding: utf-8 -*-
from __future__ import print_function, division

"""
Pytorch useful tools.
"""

import torch
import os
import errno
import numpy as np

import multiprocessing
from joblib import Parallel, delayed
import torch.nn.functional as F

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor




def save_checkpoint(state, directory, file_name):

    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, file_name + '.pth')
    torch.save(state, checkpoint_file)


def load_checkpoint(model_file):
    if os.path.isfile(model_file):
        print("=> loading model '{}'".format(model_file))
        checkpoint = torch.load(model_file)
        # print("=> loaded model '{}' (epoch {}, map {})".format(model_file, checkpoint['epoch'], checkpoint['best_map']))
        return checkpoint
    else:
        print("=> no model found at '{}'".format(model_file))
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), model_file)

def rec(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    re = len(act_set & pred_set) / len(act_set)
    return re

def prec(actual, predicted, k):
    act_set = set(actual)
    if k is not None:
        pred_set = set(predicted[:k])
    else:
        pred_set = set(predicted)
    pr = len(act_set & pred_set) / min(len(act_set), len(pred_set))
    return pr

def precak(sim, str_sim, k=None):
    act_lists = [np.nonzero(s)[0] for s in str_sim]
    pred_lists = np.argsort(-sim, axis=1)
    num_cores = min(multiprocessing.cpu_count(), 8)
    nq = len(act_lists)
    preck = Parallel(n_jobs=num_cores)(delayed(prec)(act_lists[iq], pred_lists[iq], k) for iq in range(nq))
    reck = Parallel(n_jobs=num_cores)(delayed(rec)(act_lists[iq], pred_lists[iq], k) for iq in range(nq))
    return np.mean(preck), reck



def get_graph(a, b, dist='euclidean', alpha=0.2, graph_type='propagator'): #propagator | ajacency
    weights = get_adjacency(a, b, dist=dist, alpha=alpha).float() # mask
    if graph_type == 'adjacency':
        adj = F.normalize(weights, p=1, dim=1)
        return adj
    elif graph_type == 'propagator':
        n = weights.shape[1]
        identity = torch.eye(n, dtype=weights.dtype).type(FloatTensor)
        isqrt_diag = 1. / torch.sqrt(1e-4 + torch.sum(weights, dim=-1))
        # checknan(laplacian=isqrt_diag)
        S = weights * isqrt_diag[None, :] * isqrt_diag[:, None]
        # checknan(normalizedlaplacian=S)
        propagator = identity - alpha * S
        propagator = torch.inverse(propagator[None, ...])[0]
        # checknan(propagator=propagator)
        return propagator
    else:
        return None

def get_adjacency(a,b,dist='euclidean',alpha=0.2):
    dist_map = get_dist_map(a,b,dist=dist)
    mask = dist_map != 0
    rbf_scale = 1
    weights = torch.exp(- dist_map * rbf_scale / dist_map[mask].std())
    mask = torch.eye(weights.size(1)).type(FloatTensor)
    weights = weights * (1-mask) #~mask
    return weights

def get_dist_map(a, b, dist='euclidean'):
    if dist == 'abs':
        dist_map = torch.cdist(a, b, p=1)
    elif dist == 'euclidean':
        dist_map = torch.cdist(a, b, p=2)
    elif dist == 'cosine':
        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]
        dist_map = 1 - torch.mm(a_norm, b_norm.transpose(0, 1))
    elif dist == 'cosine_sim':
        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]
        dist_map = torch.mm(a_norm, b_norm.transpose(0, 1))
    else:
        raise Exception("Distance NOT defined!")
    return dist_map



def adjust_learning_rate(args, optimizer, epoch, adj_type='continous'): #'milestone'
    """
        Updates the learning rate given an schedule and a gamma parameter.
    """
    if adj_type == 'milestone':
        if epoch in args.schedule:
            args.learning_rate *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate
    else:
        new_lr = args.learning_rate / pow((1 + 10 * epoch / args.epochs), 0.75)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
# -*- coding: utf-8 -*- 
from __future__ import print_function, division

"""
Doodle to Search
"""

# Python modules
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn

import glob
import numpy as np
import time
import os


# Own modules
from options import Options
from Logger import LogMetric
from utils import save_checkpoint, load_checkpoint, adjust_learning_rate
from test import test, test_pcyc
from train import train
from models.encoder import EncoderCNN
from models.encoder import FC_Vis
from data.generator_train import load_data
from loss.loss import DetangledJoinDomainLoss
from data.domainnet_data_prep import *
import json
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def main(args):
    print('Prepare data')
    src = args.src
    tgt = args.tgt
    print("Source: ", src, " --> Target: ", tgt)
    if args.task == 'zsl':
        train_data, [valid_sk_data, valid_im_data], [test_sk_data, test_im_data], dict_class = prepare_data(
        args=args, src=src, tgt=tgt, task=args.task)
    elif args.task == 'gzsl':
        train_data, [valid_sk_data, valid_im_data], [test_sk_seen_data, test_sk_data, test_im_data], dict_class = prepare_data(
            args=args, src=src, tgt=tgt, task=args.task)
    else:
        print("Task not defined! Give 'zsl' or 'gzsl")


    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True)
    test_sk_loader = DataLoader(test_sk_data, batch_size=3*args.batch_size, num_workers=args.prefetch, pin_memory=True)
    test_im_loader = DataLoader(test_im_data, batch_size=3*args.batch_size, num_workers=args.prefetch, pin_memory=True)
    if args.task == 'gzsl':
        test_sk_seen_loader = DataLoader(test_sk_seen_data, batch_size=3*args.batch_size, num_workers=args.prefetch, pin_memory=True)

    
    print('Create trainable model')
    if args.nopretrain:
        print('\t* Loading a pretrained model')
  
    im_net = FC_Vis(input_size = 2048, out_size=args.emb_size, hidden=1024)
    sk_net = FC_Vis(input_size=2048, out_size=args.emb_size, hidden=1024)
    print('Loss, Optimizer & Evaluation')
    criterion = DetangledJoinDomainLoss(emb_size=args.emb_size, w_sem=args.w_semantic, w_dom=args.w_domain, w_spa=args.w_triplet, lambd=args.grl_lambda, args=args)
    criterion.train()
    optimizer = torch.optim.Adam(list(im_net.parameters()) + list(sk_net.parameters()) + list(criterion.parameters()), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    print('Check CUDA')
    if args.cuda and args.ngpu > 1:
        print('\t* Data Parallel')
        im_net = nn.DataParallel(im_net, device_ids=list(range(args.ngpu)))
        sk_net = nn.DataParallel(sk_net, device_ids=list(range(args.ngpu)))
        criterion = nn.DataParallel(criterion, device_ids=list(range(args.ngpu)))

    if args.cuda:
        print('\t* CUDA')
        im_net, sk_net = im_net.cuda(), sk_net.cuda()
        criterion = criterion.cuda()


    best_map = 0
    early_stop_counter = 0
    start_epoch = 0
    if args.load is not None:
        print('Loading model')
        checkpoint = load_checkpoint(args.load)
        im_net.load_state_dict(checkpoint['im_state'])
        sk_net.load_state_dict(checkpoint['sk_state'])
        criterion.load_state_dict(checkpoint['criterion'])
        # start_epoch = checkpoint['epoch']
        start_epoch = 0
        best_map = checkpoint['best_map']
        map_valid = best_map
        print('Loaded model at epoch {epoch} and mAP {mean_ap}%'.format(epoch=checkpoint['epoch'],mean_ap=checkpoint['best_map']))
    print('***Train***')

    for epoch in range(start_epoch, args.epochs):
        # Update learning rate
        adjust_learning_rate(args, optimizer, epoch)

        loss_train, loss_sem, loss_dom, loss_spa = train(train_loader, [im_net, sk_net], optimizer, args.cuda, criterion, epoch, args.log_interval)
        # loss_train, loss_sem, loss_dom, loss_spa = None, None, None, None
        # map_valid = test(valid_im_loader, valid_sk_loader, [im_net, sk_net], args)
        save_checkpoint({'epoch': epoch + 1, 'im_state': im_net.state_dict(), 'sk_state': sk_net.state_dict(),
                         'criterion': criterion.state_dict(), 'valid_results_f': None,
                         'valid_results_sem': None,
                         'valid_results_vis': None, 'best_map': 0}, directory=args.save,
                        file_name='checkpoint')

        if (epoch + 1) % args.log_interval == 0:
            if args.task == 'gzsl':
                print("Test seen sketches!")
                valid_results_f_seen = test_pcyc(test_im_loader, test_sk_seen_loader, [im_net, sk_net, criterion], args,
                                       subset='seen',feat_type='f')
                valid_results_f_seen = None

            print("Test unseen sketches!")
            valid_results_f = test_pcyc(test_im_loader, test_sk_loader, [im_net, sk_net, criterion], args, subset='unseen',feat_type='f')
            valid_results_sem = None #test(test_im_loader, test_sk_loader, [im_net, sk_net, criterion], args, feat_type='sem')
            valid_results_vis = None #test(test_im_loader, test_sk_loader, [im_net, sk_net, criterion], args, feat_type='vis')
            map_valid = valid_results_f['map@all']
            save_checkpoint({'epoch': epoch + 1, 'im_state': im_net.state_dict(), 'sk_state': sk_net.state_dict(),
                             'criterion': criterion.state_dict(), 'valid_results_f': valid_results_f,
                             'valid_results_sem': valid_results_sem,
                             'valid_results_vis': valid_results_vis, 'best_map': map_valid}, directory=args.save,
                            file_name='model_best')

        
    print('***epoch 500 test ***')
    epoch = 500
    if args.task == 'gzsl':
        print("Test seen sketches!")
        valid_results_f_seen = test_pcyc(test_im_loader, test_sk_seen_loader, [im_net, sk_net, criterion], args,
                                subset='seen',feat_type='f')
        valid_results_f_seen = None

    print("Test unseen sketches!")
    valid_results_f = test_pcyc(test_im_loader, test_sk_loader, [im_net, sk_net, criterion], args, subset='unseen',feat_type='f')
    valid_results_sem = None #test(test_im_loader, test_sk_loader, [im_net, sk_net, criterion], args, feat_type='sem')
    valid_results_vis = None #test(test_im_loader, test_sk_loader, [im_net, sk_net, criterion], args, feat_type='vis')
    map_valid = valid_results_f['map@all']
    save_checkpoint({'epoch': epoch + 1, 'im_state': im_net.state_dict(), 'sk_state': sk_net.state_dict(),
                        'criterion': criterion.state_dict(), 'valid_results_f': valid_results_f,
                        'valid_results_sem': valid_results_sem,
                        'valid_results_vis': valid_results_vis, 'best_map': map_valid}, directory=args.save,
                    file_name='model_best')
    


if __name__ == '__main__':
    print("Start")
    # Parse options
    args = Options().parse()
    print('Parameters:\t' + str(args))

    
    # Check cuda & Set random seed
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.exp_idf is not None:
        if not os.path.isdir(os.path.join('./checkpoint', args.exp_idf)):
            os.makedirs(os.path.join('./checkpoint', args.exp_idf))
        args.save = os.path.join('./checkpoint', args.exp_idf)
        args.log = os.path.join('./checkpoint', args.exp_idf)+'/'
    if args.log is not None:
        print('Initialize logger')
        log_dir = args.log + '{}_run-batchSize_{}/' \
                .format(len(glob.glob(args.log + '*_run-batchSize_{}'.format(args.batch_size))), args.batch_size)

        args.save = log_dir
        # Create logger
        print('Log dir:\t' + log_dir)
        logger = LogMetric.Logger(log_dir, force=True)
        with open(os.path.join(args.save, 'params.txt'), 'w') as fp:
            for key, val in vars(args).items():
                fp.write('{} {}\n'.format(key, val))


    ''' Change settings and parameters here'''

    # ------------------------------------------------------------------------
    args.dataset = 'domainnet'
    args.data_path = '../../../Dataset/DomainNet/'

    sk_domains = ['sketch', 'quickdraw']
    im_domains = ['real', 'infograph', 'clipart', 'painting']

    args.src = sk_domains[0]
    args.tgt = im_domains[2]
    args.task = 'gzsl' # 'gzsl'

    args.log_interval = 500
    args.epochs = 1000
    args.batch_size = 512
    args.emb_size = 512  # 512  (em-256 / sem-300)
    args.sem_size = 300 #256 (em-256 / sem-256)
    args.lr = 1e-4
    args.schedule = []  # [10, 40]
    args.attn = True  # Default:False, try True
    # args.load = args.save +'model_best.pth'
    for param in [0.5, 1, 5]: # 0.01, 0.05, 0.1, 
        args.w_semantic = param # L_sem
        args.w_domain = 1 # L_dis
        args.w_triplet = 1 # L_vis
        param_filename = 'w_vis_' + str(args.w_triplet) + '_w_sem_' + str(args.w_semantic) + '_w_dis_' + str(args.w_domain)
        args.save = './Checkpoints/DomainNet/' + args.src + '_' + args.tgt + '_' + args.task + '/' + param_filename + '/'
        
        if not os.path.isdir(args.save):
            print("save path: ", args.save)
            os.makedirs(args.save)
        else:
            print("Existed save path: ", args.save)
        # ------------------------------------------------------------------------
        main(args)

    for param in [0.01, 0.05, 0.1, 0.5, 1, 5]: # 1
        args.w_semantic = 1 # L_sem
        args.w_domain = param # L_dis
        args.w_triplet = 1 # L_vis
        param_filename = 'w_vis_' + str(args.w_triplet) + '_w_sem_' + str(args.w_semantic) + '_w_dis_' + str(args.w_domain)
        args.save = './Checkpoints/DomainNet/' + args.src + '_' + args.tgt + '_' + args.task + '/' + param_filename + '/'
        if not os.path.isdir(args.save):
            print("save path: ", args.save)
            os.makedirs(args.save)
        else:
            print("Existed save path: ", args.save)
        # ------------------------------------------------------------------------
        main(args)

    for param in [0.01, 0.05, 0.1, 0.5, 1, 5]: # 1
        args.w_semantic = 1 # L_sem
        args.w_domain = 1 # L_dis
        args.w_triplet = param # L_vis
        param_filename = 'w_vis_' + str(args.w_triplet) + '_w_sem_' + str(args.w_semantic) + '_w_dis_' + str(args.w_domain)
        args.save = './Checkpoints/DomainNet/' + args.src + '_' + args.tgt + '_' + args.task + '/' + param_filename + '/'
        if not os.path.isdir(args.save):
            print("save path: ", args.save)
            os.makedirs(args.save)
        else:
            print("Existed save path: ", args.save)
        # ------------------------------------------------------------------------
        main(args)


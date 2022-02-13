# -*- coding: utf-8 -*-
"""
    Parse input arguments
"""

import argparse


class Options():

    def __init__(self, test=False):
        # MODEL SETTINGS
        parser = argparse.ArgumentParser(description='Zero-shot Sketch Based Retrieval',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # Positional arguments
        parser.add_argument('--dataset', type=str, default='sketchy_extend',
                            # choices=['sketchy_extend', 'tuberlin_extend',
                            #          'quickdraw_extend'],
                            help="Choose between (Sketchy).[sketchy_extend, tuberlin_extend,quickdraw_extend]")
        parser.add_argument('--train_nc', type=int, default=300, help='[DomainNet 300]')
        parser.add_argument('--test_nc', type=int, default=45, help='[Sketchy:21]')
        parser.add_argument('--n_classes', type=int, default=345, help='[Sketchy:125]')
        parser.add_argument('--src', type=str, default='sketch', help='[sketch | quickdraw]')
        parser.add_argument('--tgt', type=str, default='clipart', help='[clipart, real, painting, infograph]')
        parser.add_argument('--task', type=str, default='zsl', help='[zsl | gzsl]')
        # Model parameters
        parser.add_argument('--data_path', '-dp', type=str, default='../../../Dataset/SBIR/', help='Dataset root path.')
        parser.add_argument('--emb_size', type=int, default=256, help='Embedding Size.')
        parser.add_argument('--sem_size', type=int, default=300, help='Semantic Embedding Size.')
        parser.add_argument('--grl_lambda', type=float, default=0.5, help='Lambda used to normalize the GRL layer.')
        parser.add_argument('--nopretrain', action='store_false', help='Loads a pretrained model (Default: True).')
        # Optimization options
        parser.add_argument('--epochs', '-e', type=int, default=1000, help='Number of epochs to train.')
        parser.add_argument('--batch_size', '-b', type=int, default=20, help='Batch size.')
        parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='The Learning Rate.')
        parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
        parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
        parser.add_argument('--schedule', type=int, nargs='+', default=[],
                            help='Decrease learning rate at these epochs.')
        parser.add_argument('--w_semantic', type=float, default=1, help='Semantic loss Weight.')
        parser.add_argument('--w_domain', type=float, default=1, help='Domain loss Weight.')
        parser.add_argument('--w_triplet', type=float, default=1, help='Triplet loss Weight.')
        parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
        parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        parser.add_argument('--save', '-s', type=str, default=None, help='Folder to save checkpoints.')
        parser.add_argument('--load', '-l', type=str, default=None, help='Checkpoint path to resume / test.')
        parser.add_argument('--early_stop', '-es', type=int, default=200, help='Early stopping epochs.')
        # Acceleration
        parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU, 1 = CUDA, 1 < DataParallel')
        parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
        # i/o
        parser.add_argument('--log', type=str, default=None, help='Log folder.')
        parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                            help='How many batches to wait before logging training status')
        parser.add_argument('--attn', action='store_true', help='Attention module (Default: False).')
        parser.add_argument('--plot', action='store_true', help='Qualitative results (Default: False).')
        parser.add_argument('--exp_idf', type=str,  default=None, help='Provide an exp_idf for logging purpose.')

        # Test
        if test:
            parser.add_argument('--num_retrieval', type=int, default=10, help='Number of images to be retrieved')
        self.parser = parser

    def parse(self):

        args = self.parser.parse_args()
        return args


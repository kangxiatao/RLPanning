# -*- coding: utf-8 -*-

"""
Created on 03/23/2022
configs.
@author: AnonymousUser314156
"""

import os
import time
import argparse
from utils import mail_log
from easydict import EasyDict as edict
import logging
from tensorboardX import SummaryWriter


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int)  # Sets PyTorch and Numpy seeds
    ## Training Hyperparameters ##
    parser.add_argument('--config', type=str, default='mnist/lenet5/99.99',
                        help='config (dataset/model(network depth)/pruning rate(percentage)) - e.g.[cifar10/vgg19/98, mnist/lenet5/90]')
    parser.add_argument('--pretrained', type=str, default='666',
                        help='path - runs/anoi/finetune_cifar10_vgg19_l2_best.pth.tar')
    parser.add_argument('--run', type=str, default='test_exp', help='experimental notes (default: test_exp)')
    parser.add_argument('--epoch', type=int, default=666)
    parser.add_argument('--batch_size', type=int, default=666)
    parser.add_argument('--l2', type=str, default=666)
    parser.add_argument('--lr_mode', type=str, default='cosine', help='cosine or preset')
    parser.add_argument('--optim_mode', type=str, default='SGD', help='SGD or Adam')
    parser.add_argument('--storage_mask', type=int, default=0)  # storage mask
    parser.add_argument('--debug', type=str, default='0')  # Debug flags (printing data, plotting, etc.)
    parser.add_argument('--dp', type=str, default='../Data', help='dataset path')
    ## RL Hyperparameters ##
    parser.add_argument("--policy", default="TD3", type=str)  # Policy name (TD3, DDPG)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--start_timesteps", default=2e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=2e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=5e4, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--panning_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.01)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--eval_mode", default=0, type=int)  # Enable evaluation mode
    parser.add_argument("--save_model", default=0, type=int)  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    # parser.add_argument("--load_model", default="default")  # Model load file name, "" doesn't load, "default" uses file_name
    ## Pruning Hyperparameters ##
    parser.add_argument('--rank_algo', type=str, default='snip',
                        help='rank algorithm (choose one of: snip, grasp, synflow)')
    parser.add_argument('--prune_mode', type=str, default='panning',
                        help='prune mode (choose one of: dense, rank, rank/random, rank/iterative, fcs, panning)')
    parser.add_argument('--num_iters_prune', type=int, default=100)
    args = parser.parse_args()

    ## Default Config ##
    exp_set = args.config.split('/')  # dataset model prune_ratio
    exp_name = f'{exp_set[0]}_{exp_set[1]}_prune{exp_set[2]}'.replace('.', '_')
    base_config = {'network': "vgg", 'depth': 19, 'dataset': 'cifar10',
                   'batch_size': 128, 'epoch': 180, 'learning_rate': 0.1, 'weight_decay': 5e-4,
                   'target_ratio': 0.90, 'samples_per_class': 10,
                   'train_mode': 1, 'dynamic': 0, 'data_mode': 0, 'num_group': 1}
    config = edict(base_config)
    config.dataset = exp_set[0]
    config.target_ratio = float(exp_set[2])/100.0
    config.network = ''.join(list(filter(str.isalpha, exp_set[1])))
    config.depth = int(''.join(list(filter(str.isdigit, exp_set[1]))))
    if 'mnist' in config.dataset:
        config.batch_size = 256
        config.epoch = 80
        config.weight_decay = 1e-4
        config.classe = 10
    elif 'cifar' in config.dataset:
        config.batch_size = 128
        config.epoch = 180
        config.weight_decay = 5e-4
        if config.dataset == 'cifar10':
            config.classe = 10
        else:
            config.classe = 100
            if 'resnet' in config.network:
                config.samples_per_class = 5
                config.num_iters = 2
    elif 'imagenet' in config.dataset:
        config.batch_size = 128
        config.epoch = 300
        config.classe = 200
        if 'vgg' in config.network:
            config.weight_decay = 5e-4
            config.samples_per_class = 5
            config.num_iters = 2
        elif 'resnet' in config.network:
            config.weight_decay = 1e-4
            config.samples_per_class = 1
            config.num_iters = 10
    # config.num_iters_prune = 100
    config.num_iters_prune = args.num_iters_prune

    ## Experiment Name and Out Path ##
    summn = [exp_name]
    chekn = [exp_name]
    if len(args.run) > 0:
        summn.append(args.run)
        chekn.append(args.run)
        config.exp_name = exp_name + '_' + args.run
    _str = ''.join(list(filter(str.isalpha, args.prune_mode)))
    summn[-1] += f'_{_str}'
    chekn[-1] += f'_{_str}'
    config.exp_name += f'_{_str}'
    if args.rank_algo != '666' and ('dense' not in args.prune_mode and 'panning' not in args.prune_mode):
        summn[-1] += f'_{args.rank_algo}'
        chekn[-1] += f'_{args.rank_algo}'
        config.exp_name += f'_{args.rank_algo}'
    summn.append("summary/")
    chekn.append("checkpoint/")
    summary_dir = ["./runs/pruning"] + exp_set[:-1] + summn
    ckpt_dir = ["./runs/pruning"] + exp_set[:-1] + chekn
    config.summary_dir = os.path.join(*summary_dir)
    config.checkpoint_dir = os.path.join(*ckpt_dir)
    print("=> config.summary_dir:    %s" % config.summary_dir)
    print("=> config.checkpoint_dir: %s" % config.checkpoint_dir)

    ## Console Parameters ##
    if args.pretrained != '666':
        config.pretrained = args.pretrained
        print("use pre-trained mode:{}".format(config.pretrained))
    else:
        config.pretrained = None
    if args.epoch != 666:
        config.epoch = args.epoch
        print("set new epoch:{}".format(config.epoch))
    if args.batch_size != 666:
        config.batch_size = args.batch_size
        print("set new batch_size:{}".format(config.batch_size))
    if args.l2 != 666:
        config.weight_decay = args.l2
        print("set new weight_decay:{}".format(config.weight_decay))
    config.prune_mode = args.prune_mode
    config.storage_mask = args.storage_mask
    config.lr_mode = args.lr_mode
    config.optim_mode = args.optim_mode
    config.debug = args.debug
    config.dp = args.dp
    config.send_mail_head = (config.exp_name + ' -> ' + args.run + '\n')
    config.send_mail_str = (mail_log.get_words() + '\n')
    if args.rank_algo != '666':
        # grasp: --grad_mode 0 --score_mode 1
        # snip: --grad_mode 3 --score_mode 2
        if args.rank_algo.lower() == 'grasp':
            config.grad_mode = 0
            config.score_mode = 2
        elif args.rank_algo.lower() == 'snip':
            config.grad_mode = 3
            config.score_mode = 2
        elif args.rank_algo.lower() == 'grasp_ori':
            config.data_mode = 9
            config.grad_mode = 0
            config.score_mode = 1
        elif args.rank_algo.lower() == 'snip_ori':
            config.data_mode = 9
            config.grad_mode = 3
            config.score_mode = 2
        elif args.rank_algo.lower() == 'synflow':
            pass
        else:
            print("choose one of: GraSP, SNIP, SynFlow")
            pass
        config.rank_algo = args.rank_algo
        print("set pruning algorithm: {}".format(args.rank_algo))
    # Foresight Connection Sensitivity
    if 'fcs' in config.prune_mode:
        config.dynamic = 1
        config.data_mode = 9
        config.grad_mode = 3
        config.score_mode = 2
        config.prune_mode += '_rank_iterative'
    if 'panning' in config.prune_mode:
        config.dynamic = 1

    if 'exp' in config.exp_name:
        config.epoch = 0

    config.seed = args.seed
    config.lr = args.lr
    config.policy = args.policy
    config.start_timesteps = args.start_timesteps
    config.eval_freq = args.eval_freq
    config.max_timesteps = args.max_timesteps
    config.expl_noise = args.expl_noise
    config.panning_size = args.panning_size
    config.discount = args.discount
    config.tau = args.tau
    config.policy_noise = args.policy_noise
    config.noise_clip = args.noise_clip
    config.policy_freq = args.policy_freq
    config.eval_mode = args.eval_mode
    config.save_model = args.save_model
    config.load_model = args.load_model

    return config


def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


def get_logger(name, logpath, displaying=True, saving=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    log_path = logpath + name + time.strftime("-%Y%m%d-%H%M%S")
    makedirs(log_path)
    if saving:
        info_file_handler = logging.FileHandler(log_path, encoding='utf-8')
        info_file_handler.setLevel(logging.INFO)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger


def init_logger(config):
    makedirs(config.summary_dir)
    makedirs(config.checkpoint_dir)

    # set logger
    logger = get_logger('log', logpath=config.summary_dir + '/')
    logger.info(dict(config))
    writer = SummaryWriter(config.summary_dir)
    return logger, writer
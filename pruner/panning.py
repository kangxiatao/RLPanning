# -*- coding: utf-8 -*-

"""
Created on 05/10/2022
panning.
@author: AnonymousUser314156
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

import os
import copy
import random
import datetime
import types
from tqdm import tqdm

from models import DDPG
from models import TD3
from utils import buffer_utils

from utils.prune_utils import reset_mask, fetch_data, linearize, nonlinearize, get_keep_ratio, FetchData
from utils.matplot_utils import PlotPanning

from tensorboardX import SummaryWriter


def normalization(scores):
    all_scores = torch.cat([torch.flatten(x) for x in scores.values()])
    mean_div = all_scores.mean()
    all_scores.div_(mean_div)
    std = all_scores.std()
    mean = all_scores.mean()
    # print(mean, std)
    for m, s in scores.items():
        scores[m] = ((scores[m]/mean_div - mean) / std).abs_()
    return scores


def get_loss_gtg(model, loader, masks=None):
    old_modules = list(model.modules())
    net = copy.deepcopy(model)
    net.train()
    net.zero_grad()
    if masks:
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight.data = layer.weight.data * masks[old_modules[idx]]
    weights = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weights.append(layer.weight)
    for w in weights:
        w.requires_grad_(True)
    inputs, targets = loader
    _outputs = net.forward(inputs) / 200
    _loss = F.cross_entropy(_outputs, targets)
    _grad = autograd.grad(_loss, weights, create_graph=True)
    _gtg, _layer = 0, 0
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            _g = _grad[_layer]
            _gtg += _g.pow(2).sum()
            _layer += 1
    return _loss.cpu().detach().numpy(), _gtg.cpu().detach().numpy()


class PruningEnv:
    """
    Environment for pruning
    """
    def __init__(self, model, trainloader, device, config):

        self.trian_cnt = 1
        self.debug = config.debug
        self.render = False
        self.ratio = config.target_ratio
        self.old_net = model
        self.m = list(model.modules())
        self.net = copy.deepcopy(model)
        self.masks = reset_mask(model)
        self.fd = FetchData(trainloader, device, config.classe, config.samples_per_class, config.data_mode)
        self.num_iters_prune = config.num_iters_prune

        self.weights = dict()
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                self.weights[layer] = torch.clone(layer.weight.data)
        self.old_weights = []
        for layer in self.old_net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                self.old_weights.append(layer.weight)
        for w in self.old_weights:
            w.requires_grad_(True)

        self.init_loss = 0
        self.init_gtg = 0
        for epoch in range(10):  # init
            _loss, _gtg = get_loss_gtg(model, self.fd.samples())
            self.init_loss += _loss
            self.init_gtg += _gtg
            if epoch == 0:
                self.max_loss, self.min_loss = _loss / 100, _loss * 100  # for normalization
                self.max_gtg, self.min_gtg = _gtg / 100, _gtg * 100
            else:
                if _loss > self.max_loss: self.max_loss = _loss
                if _loss < self.min_loss: self.min_loss = _loss
                if _gtg > self.max_gtg: self.max_gtg = _gtg
                if _gtg < self.min_gtg: self.min_gtg = _gtg
        self.init_loss /= 10
        self.init_gtg /= 10
        self.reset()

        if 'fig' in self.debug:
            self.fig = PlotPanning(600, 400, 60)

    def step(self, action):
        """
        action: (SynFlow, SNIP, GraSP, Ratio) ACTION SPACE [0,1]
        """

        # === Calculate score ===
        score = dict()
        for m, s in self.s_synflow.items():
            score[m] = (action[0]+1)*self.s_synflow[m]+(action[1]+1)*self.s_snip[m]+(action[2]+1)*self.s_grasp[m]
            # score[m] = self.s_synflow[m].pow((action[0]+1)/2) * self.s_snip[m].pow((action[1]+1)/2) * self.s_grasp[m].pow((action[2]+1)/2)
        # keep_ratio = (1.0 - self.ratio)**((self.iter_cnt + 1) / self.num_iters_prune) if self.trian_cnt < 10000 else (action[3]+1)/2
        keep_ratio = (1.0 - self.ratio)**((self.iter_cnt + 1) / self.num_iters_prune)
        target_ratio = 1 - keep_ratio
        all_scores = torch.cat([torch.flatten(x) for x in score.values()])
        topk = int(len(all_scores) * (1-target_ratio))
        if topk == 0: topk = 1
        if target_ratio == 0: topk -= 1
        threshold, _index = torch.topk(all_scores, topk)
        acceptable_score = threshold[-1]
        for m, g in score.items():
            self.masks[m] = (g >= acceptable_score).float()

        # === Environment ===
        inputs, targets = self.fd.samples()
        # === Calculate SNIP, GraSP, SynFLow (g, Hg, g) ===
        _outputs = self.old_net.forward(inputs) / 200
        _loss = F.cross_entropy(_outputs, targets)
        _grad = autograd.grad(_loss, self.old_weights, create_graph=True)
        _gtg, _layer = 0, 0
        for idx, layer in enumerate(self.old_net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                _gtg += _grad[_layer].pow(2).sum()
                _layer += 1
        _Hg = autograd.grad(_gtg, self.old_weights, retain_graph=True)
        for idx, layer in enumerate(self.net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight.data = (self.weights[self.m[idx]] * self.masks[self.m[idx]]).abs_()
        self.net.zero_grad()
        _output = self.net(self.input_one)
        torch.sum(_output).backward()
        # Calculate score
        s_synflow, s_snip, s_grasp = dict(), dict(), dict()
        true_masks = dict()  # effe_ratio
        layer_cnt = 0
        for idx, layer in enumerate(self.net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                _w = self.weights[self.m[idx]]
                s_synflow[self.m[idx]] = _w * layer.weight.grad
                s_snip[self.m[idx]] = _w * _grad[layer_cnt]
                s_grasp[self.m[idx]] = _w * _Hg[layer_cnt]
                true_masks[self.m[idx]] = (layer.weight * layer.weight.grad != 0).float()
                layer_cnt += 1
        self.s_synflow = normalization(s_synflow)
        self.s_snip = normalization(s_snip)
        self.s_grasp = normalization(s_grasp)
        # === state & reward ===
        # complete loss and gtg
        _loss = _loss.cpu().detach().numpy()
        _gtg = _gtg.cpu().detach().numpy()
        if _loss > self.max_loss: self.max_loss = _loss
        if _loss < self.min_loss: self.min_loss = _loss
        if _gtg > self.max_gtg: self.max_gtg = _gtg
        if _gtg < self.min_gtg: self.min_gtg = _gtg
        # sub-net network
        sub_loss, sub_gtg = get_loss_gtg(self.old_net, (inputs, targets), self.masks)
        if sub_loss > self.max_loss: self.max_loss = sub_loss
        if sub_loss < self.min_loss: self.min_loss = sub_loss
        if sub_gtg > self.max_gtg: self.max_gtg = sub_gtg
        if sub_gtg < self.min_gtg: self.min_gtg = sub_gtg
        norm_loss = (_loss-self.min_loss)/(self.max_loss-self.min_loss)
        norm_gtg = (_gtg-self.min_gtg)/(self.max_gtg-self.min_gtg)
        r_loss = abs((sub_loss-_loss)/(self.max_loss-self.min_loss))
        r_gtg = abs((sub_gtg-_gtg)/(self.max_gtg-self.min_gtg)) / 5
        effe_ratio = 1-get_keep_ratio(true_masks)
        r_ratio = abs(target_ratio - effe_ratio) * 100
        # --- loss value, gradient norm, compression condition, target compression, number of iterations ---
        next_state = (norm_loss, norm_gtg, r_loss, r_gtg, effe_ratio, target_ratio, self.iter_cnt/self.num_iters_prune)
        # --- Goal: effective compression = current desired compression = set compression, minimize impact on loss and gtg ---
        reward = - r_ratio - r_loss - r_gtg
        # === done ===
        if effe_ratio == 1:  # Effective compression to 100% model is dead, game over
            done = True
            reward -= (1 - self.iter_cnt / self.num_iters_prune) * (self.num_iters_prune)
        else:
            if self.iter_cnt + 1 >= self.num_iters_prune: done = True
            else: done = False

        if self.render and self.iter_cnt % 20 == 0:
            print('action', action)
            print('target_ratio, effe_ratio, iter_cnt', (target_ratio, effe_ratio, self.iter_cnt))
            print('r_loss,r_gtg,r_ratio ', r_loss,r_gtg,r_ratio)
            print('reward, done', reward, done)
            print('-'*5, self.ratio)
        if self.render and 'fig' in self.debug:
            self.fig.append(action,(r_ratio,r_loss,r_gtg))
            if done:
                self.fig.plot()

        self.iter_cnt += 1

        return next_state, reward, done, 0

    def action_sample(self):
        a = random.uniform(-1, 1)
        b = random.uniform(-1, 1)
        c = random.uniform(-1, 1)
        # d = random.uniform(-1, 1)
        return a, b, c

    def reset(self):
        """
        Reset the model and mask
        Recalculate the multi-metric score of the whole network
        """
        self.net = copy.deepcopy(self.old_net)
        self.masks = reset_mask(self.old_net)
        th = random.uniform(0, 1)
        if self.trian_cnt > 10000:
            self.ratio = random.uniform(0.9, 1) if th > 0.5 else random.uniform(0.98, 1)
        else:
            self.ratio = random.uniform(0.8, 1)

        self.old_net.train()
        self.old_net.zero_grad()
        self.net.eval()
        self.net.zero_grad()
        signs = linearize(self.net)

        inputs, targets = self.fd.samples()
        input_dim = list(inputs[0, :].shape)
        self.input_one = torch.ones([1] + input_dim).to(self.fd.device)
        self.iter_cnt = 0

        # === Calculate SNIP, GraSP, SynFLow (g, Hg, g) ===
        _outputs = self.old_net.forward(inputs) / 200
        _loss = F.cross_entropy(_outputs, targets)
        _grad = autograd.grad(_loss, self.old_weights, create_graph=True)
        _gtg, _layer = 0, 0
        for idx, layer in enumerate(self.net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                _g = _grad[_layer]
                _gtg += _g.pow(2).sum()
                _layer += 1
        _Hg = autograd.grad(_gtg, self.old_weights, retain_graph=True)
        _output = self.net(self.input_one)
        torch.sum(_output).backward()
        # === Calculate score ===
        s_synflow, s_snip, s_grasp = dict(), dict(), dict()
        layer_cnt = 0
        for idx, layer in enumerate(self.net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                _w = self.weights[self.m[idx]]
                s_synflow[self.m[idx]] = _w * layer.weight.grad
                s_snip[self.m[idx]] = _w * _grad[layer_cnt]
                s_grasp[self.m[idx]] = _w * _Hg[layer_cnt]
                layer_cnt += 1
        self.s_synflow = normalization(s_synflow)
        self.s_snip = normalization(s_snip)
        self.s_grasp = normalization(s_grasp)

        _loss = _loss.cpu().detach().numpy()
        _gtg = _gtg.cpu().detach().numpy()
        norm_loss = (_loss-self.min_loss)/(self.max_loss-self.min_loss)
        norm_gtg = (_gtg-self.min_gtg)/(self.max_gtg-self.min_gtg)
        state = (norm_loss, norm_gtg, 0, 0, 0, 0, 0)  #
        return state


# Runs policy for X episodes and returns average reward
def eval_policy(policy, eval_env, eval_episodes=1):
    eval_env.render = True
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
        state, done = eval_env.reset(), False

    avg_reward /= eval_episodes
    eval_env.render = False

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def RLPanning(model, trainloader, device, config):

    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(curr_time)

    env = PruningEnv(model, trainloader, device, config)

    file_name = f"{config.policy}_{config.seed}_test"
    print("---------------------------------------")
    print(f"Policy: {config.policy}, Seed: {config.seed}")
    print("---------------------------------------")

    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if config.save_model and not os.path.exists("./runs/policy/"):
        os.makedirs("./runs/policy/")

    state_dim = 7
    action_dim = 3
    max_action = 1

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": config.discount,
        "tau": config.tau,
    }

    writer = SummaryWriter('./runs/policy/')

    # Initialize policy
    if config.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = config.policy_noise * max_action
        kwargs["noise_clip"] = config.noise_clip * max_action
        kwargs["policy_freq"] = config.policy_freq
        kwargs["lr"] = config.lr
        kwargs["writer"] = writer
        policy = TD3.TD3(**kwargs)
    elif config.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    if config.load_model != "":
        policy_file = file_name if config.load_model == "default" else config.load_model
        policy.load(f"./runs/policy_exp/{policy_file}")

    if config.eval_mode:
        # eval_policy(policy, env)
        pass
    else:
        replay_buffer = buffer_utils.ReplayBuffer(state_dim, action_dim)

        # Evaluate untrained policy
        evaluations = [eval_policy(policy, env, 1)]

        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        for t in range(int(config.max_timesteps)):

            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < config.start_timesteps:
                action = env.action_sample()
            else:
                action = (
                        policy.select_action(np.array(state))
                        + np.random.normal(0, max_action * config.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)

            # Perform action
            next_state, reward, done, _ = env.step(action)
            env.trian_cnt += 1
            # done_bool = float(done) if episode_timesteps < env.num_iters_prune else 0
            done_bool = float(done)

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= config.start_timesteps:
                policy.train(replay_buffer, config.panning_size)

            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                if episode_num % 5 == 0:
                    print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                # Reset environment
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Evaluate episode
            if (t + 1) % config.eval_freq == 0:
                avg_reward = eval_policy(policy, env)
                writer.add_scalar('td3/avg_reward', avg_reward, int((t + 1) / config.eval_freq))
                evaluations.append(avg_reward)
                if config.save_model:
                    np.save(f"./runs/policy/{file_name}", evaluations)
                    policy.save(f"./runs/policy/{file_name}")

    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(curr_time)

    state, done = env.reset(), False
    env.ratio = config.target_ratio
    env.render = True
    while not done:
        action = policy.select_action(np.array(state))
        state, reward, done, _ = env.step(action)

    return env.masks, 0


def Panning(model, dataloader, device, config):

    psta = 0
    if 'mul' in config.debug:
        pstep = [[0.3, 1, 0.2], [0.3, 1, 0.4], [0.5, 0.5, 0.6], [0.8, 0, 1], [1, 0, 1]]
    else:
        pstep = [[0.2, 0.5, 0.3], [0.2, 0.4, 0.4], [0.2, 0.3, 0.5], [0.4, 0.2, 0.4], [0.5, 0.0, 0.5]]
    stair = [0.8, 0.9, 0.98, 0.99, 1]

    @torch.no_grad()
    def linearize(model):
        signs = {}
        for name, param in model.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(model, signs):
        for name, param in model.state_dict().items():
            param.mul_(signs[name])

    old_net = model
    net = copy.deepcopy(model)
    signs = linearize(net)
    net.eval()
    net.zero_grad()
    modules_ls = list(old_net.modules())
    (data, _) = next(iter(dataloader))
    input_dim = list(data[0, :].shape)
    input = torch.ones([1] + input_dim).to(device)
    copy_net_weights = dict()
    for layer in old_net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            copy_net_weights[layer] = torch.clone(layer.weight.data)

    old_net.train()
    old_net.zero_grad()
    weights = []
    for layer in old_net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            weights.append(layer.weight)
    for w in weights:
        w.requires_grad_(True)

    keep_masks = reset_mask(old_net)
    score = dict()

    ratio = config.target_ratio
    num_iters = config.num_iters_prune
    desc = ('[r=%s] s: %e' % (1, 0))
    prog_bar = tqdm(range(num_iters), total=num_iters, desc=desc, leave=True) if num_iters > 1 else range(num_iters)
    for epoch in prog_bar:
        keep_ratio = (1.0 - ratio)**((epoch + 1) / num_iters)

        inputs, targets = fetch_data(dataloader, config.classe, config.samples_per_class, dm=config.data_mode)
        inputs = inputs.to(device)
        targets = targets.to(device)

        # === Calculate SNIP and GraSP (g, Hg) ===
        _outputs = old_net.forward(inputs) / 200
        _loss = F.cross_entropy(_outputs, targets)
        _grad = autograd.grad(_loss, weights, create_graph=True)
        _gtg, _layer = 0, 0
        for idx, layer in enumerate(old_net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                _g = _grad[_layer]
                _gtg += _g.pow(2).sum()
                _layer += 1
        _Hg = autograd.grad(_gtg, weights, retain_graph=True)

        # === Calculate SynFlow (g) ===
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight.data = (copy_net_weights[modules_ls[idx]] * keep_masks[modules_ls[idx]]).abs_()
        net.zero_grad()
        _output = net(input)
        torch.sum(_output).backward()

        # === Calculate score ===
        score = dict()
        s_synflow = dict()
        s_snip = dict()
        s_grasp = dict()
        layer_cnt = 0
        for idx, layer in enumerate(net.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                _w = copy_net_weights[modules_ls[idx]]
                s_synflow[modules_ls[idx]] = _w * layer.weight.grad
                s_snip[modules_ls[idx]] = _w * _grad[layer_cnt]
                s_grasp[modules_ls[idx]] = _w * _Hg[layer_cnt]
                layer_cnt += 1

        if stair[psta] < (1-keep_ratio):
            psta += 1

        s_synflow = normalization(s_synflow)
        s_snip = normalization(s_snip)
        s_grasp = normalization(s_grasp)
        for m, s in s_synflow.items():
            if 'mul' in config.debug:
                score[m] = s_synflow[m].pow(pstep[psta][0])*s_snip[m].pow(pstep[psta][1])*s_grasp[m].pow(pstep[psta][2])
            else:
                score[m] = pstep[psta][0]*s_synflow[m]+pstep[psta][1]*s_snip[m]+pstep[psta][2]*s_grasp[m]

        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for x in score.values()])
        threshold, _index = torch.topk(all_scores, int(len(all_scores) * keep_ratio))
        acceptable_score = threshold[-1]
        for m, g in score.items():
            keep_masks[m] = (g >= acceptable_score).float()

        if num_iters > 1:
            desc = ('[r=%s] s: %e' % (keep_ratio, acceptable_score.cpu().detach().numpy()))
            prog_bar.set_description(desc, refresh=True)

    # nonlinearize(net, signs)

    return keep_masks, score


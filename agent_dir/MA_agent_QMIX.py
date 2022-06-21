import os
import random
import copy
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
from agent_dir.agent import Agent
from copy import deepcopy
from .utils import *

from torchvision.models import resnet, alexnet, vgg
from loguru import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):

    def __init__(self, state_size, action_size):
        super(QNet, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.LayerNorm(512),
            nn.SiLU(),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
        )

        self.lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True)

        self.head_V = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 1),
        )
        self.head_Adv = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, action_size),
        )

    def forward(self, x):
        self.lstm.flatten_parameters()

        x = self.base(x)
        x, (h, c) = self.lstm(x)

        advantage = self.head_Adv(x)
        value = self.head_V(x)
        return value + advantage - advantage.mean()

class MIXNet(nn.Module):
    def __init__(self, n_agent, state_dim):
        super(MIXNet, self).__init__()

        self.n_agent = n_agent
        self.state_dim = state_dim
        self.mixing_hidden_size = 512

        # Used to generate mixing network
        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.state_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, self.n_agent * self.mixing_hidden_size)
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.state_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, self.mixing_hidden_size)
        )

        self.hyper_b1 = nn.Linear(self.state_dim, self.mixing_hidden_size)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.state_dim, self.mixing_hidden_size),
            nn.LayerNorm(self.mixing_hidden_size),
            nn.SiLU(),
            nn.Linear(self.mixing_hidden_size, 1)
        )

        self.ln1=nn.LayerNorm(self.mixing_hidden_size)

    def forward(self, q_all, s_global):
        B, ep_len=q_all.shape[:2]
        q_all=q_all.unsqueeze(2)

        w1 = torch.abs(self.hyper_w1(s_global))  # (batch_size, max_episode_len, N * qmix_hidden_dim)
        b1 = self.hyper_b1(s_global)  # (batch_size, max_episode_len, qmix_hidden_dim)
        w1 = w1.view(B, ep_len, self.n_agent, self.mixing_hidden_size)  # (batch_size, max_episode_len, N,  qmix_hidden_dim)
        b1 = b1.view(B, ep_len, 1, self.mixing_hidden_size)  # (batch_size, max_episode_len, 1, qmix_hidden_dim)

        q_hidden = F.elu(self.ln1(q_all @ w1 + b1))  # (batch_size, max_episode_len, 1, qmix_hidden_dim)

        w2 = torch.abs(self.hyper_w2(s_global))  # (batch_size, max_episode_len, qmix_hidden_dim)
        b2 = self.hyper_b2(s_global)  # (batch_size, max_episode_len,1)
        w2 = w2.view(B, ep_len, self.mixing_hidden_size, 1)  # (batch_size, max_episode_len, qmix_hidden_dim, 1)
        b2 = b2.view(B, ep_len, 1, 1)  # (batch_size, max_episode_len, 1， 1)

        q_total = q_hidden @ w2 + b2  # (batch_size, max_episode_len, 1， 1)
        q_total = q_total.view(B, -1)  # (batch_size, max_episode_len)
        return q_total

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_eps = []
        self.buffer = []
        self.buffer_size = buffer_size

    def __len__(self):
        return len(self.buffer_eps)

    def proc(self, state, action, reward, next_state, done):
        return state.float(), action.float(), reward, next_state.float(), done

    def push(self, *transition):
        self.buffer.append(self.proc(*transition))

    def commit(self):
        data_eps = [torch.stack(x, dim=0).to(device) for x in list(zip(*self.buffer))]
        self.buffer.clear()

        if len(self.buffer_eps) >= self.buffer_size:
            self.buffer_eps.pop(0)
        self.buffer_eps.append(data_eps)

    def sample(self, batch_size):
        batch = random.sample(self.buffer_eps, batch_size)
        return [torch.stack(x, dim=0).to(device) for x in list(zip(*batch))]

    def clean(self):
        self.buffer.clear()
        self.buffer_eps.clear()


class AgentQMIX():
    def __init__(self, n_act, n_state, n_agent, mix_net, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        self.n_act = n_act
        self.n_state = n_state
        self.n_agent = n_agent

        self.Qnet = QNet(self.n_state, self.n_act).to(device)

        self.Qnet_T = deepcopy(self.Qnet).to(device)
        for m in self.Qnet_T.parameters():
            m.requires_grad = False

        self.eps = args.eps_start

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam([
                {'params': self.Qnet.parameters()},
                {'params': mix_net.parameters()}
            ], lr=args.lr, weight_decay=5e-4)

        self.args = args

        self.eps_scd = EpsScheduler(args.eps_start, args.eps_end, args.eps_decay)
        self.ema = EMA(args.ema)

    def soft_update(self):
        self.ema.update_model_average(self.Qnet_T, self.Qnet)

    def eps_update(self):
        self.eps = self.eps_scd.step()

    def train_step_Q(self, state, next_state, action):# [B,step_ep,N]
        Qi = self.Qnet(state).gather(2, action.unsqueeze(-1).long()).squeeze(-1)
        with torch.no_grad():
            Qi_T = self.Qnet_T(next_state).max(dim=-1)[0]
        return Qi, Qi_T

    def train_step_after(self, Q_mix, Q_T_mix, reward): # [B,step_ep,N]
        y = deepcopy(reward.float())
        with torch.no_grad():
            y += self.args.gamma * Q_T_mix

        loss = self.criterion(Q_mix, y)
        return loss

    def train_step_backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(parameters=self.Qnet.parameters(), max_norm=self.args.grad_norm_clip)
        self.optimizer.step()

    #@torch.no_grad()
    def make_action(self, state, test=False):
        """
        Return predicted action of your agent
        Input:observation
        Return:action
        """
        if test:
            return self.Qnet(state).argmax(dim=-1)
        else:
            return self.Qnet(state).argmax(dim=-1) if random.random() > self.eps else torch.randint(0, self.n_act, (1,))

class MA_QMIX():
    def __init__(self, env, args):
        self.env=env
        self.args=args
        self.writer = SummaryWriter("log")

        self.n_agent = len(env.action_space)
        self.n_act = env.action_space[0].n
        self.n_state = env.observation_space[0].shape[0]

        self.mix_net = MIXNet(self.n_agent, self.n_state*self.n_agent).to(device)

        self.agent_list=[AgentQMIX(self.n_act, self.n_state, self.n_agent, self.mix_net, args)]*self.n_agent

        self.mem = ReplayBuffer(args.buffer_size)

    def train_step(self, state_all, action_all, reward_all, next_state_all, done_all): # [B,step_ep,n_agent,N]
        self.agent_list[0].soft_update()

        reward_all = (reward_all+7)/10

        Q_all = []
        QT_all = []
        for i, agent in enumerate(self.agent_list):
            Qi, Qi_T = agent.train_step_Q(state_all[:, :, i, :], next_state_all[:, :, i, :], action_all[:, :, i])
            Q_all.append(Qi)
            QT_all.append(Qi_T)

        Q_all=torch.stack(Q_all, dim=2)
        QT_all=torch.stack(QT_all, dim=2)
        Q_mix=self.mix_net(Q_all, state_all.flatten(2))
        QT_mix=self.mix_net(QT_all, next_state_all.flatten(2))

        loss=[]
        for i, agent in enumerate(self.agent_list):
            loss_i = agent.train_step_after(Q_mix, QT_mix, reward_all[:,:,i])
            loss.append(loss_i)

        loss=sum(loss)/self.n_agent
        self.agent_list[0].train_step_backward(loss)

        return loss.item()

    @torch.no_grad()
    def make_action_all(self, state_all):
        return [agent.make_action(state_all[:,i,:].unsqueeze(0), self.args.test).view(-1)[-1] for i,agent in enumerate(self.agent_list)]

    def train(self):
        """
        Implement your training algorithm here
        """
        n_ep = self.args.n_ep
        step = 0
        loss_sum = 0

        for episode in range(n_ep):
            ep_r = 0
            ep_r_all = np.zeros(3)
            step_inter=0

            state_list=[]

            state_all = self.env.reset()
            state_all = torch.tensor(state_all, device=device)

            while True:
                if self.args.render:
                    self.env.render(mode=None)

                state_list.append(state_all.float())

                #action_list = [agent.make_action(state_all[i].unsqueeze(0).float(), self.args.test).detach().cpu() for i,agent in enumerate(self.agent_list)]
                action_list = self.make_action_all(torch.stack(state_list, dim=0))
                action_all = torch.tensor(action_list)

                for agent in self.agent_list:
                    agent.eps_update()

                next_state_list, reward_list, done_list, info = self.env.step([to_onehot(x.view(-1), n=self.n_act).numpy() for x in action_list])
                next_state_all=torch.tensor(next_state_list)
                reward_all=torch.tensor(reward_list)
                done_all=torch.tensor(done_list)

                ep_r += reward_all.mean()
                ep_r_all += reward_all.numpy()

                self.mem.push(*[torch.tensor(x, device='cpu') for x in [state_all.cpu(), action_all, reward_all, next_state_all, done_all]])

                step += 1
                step_inter += 1

                if step_inter>=self.args.max_step or done_all.all():
                    self.writer.add_scalar("ep_r", ep_r/step_inter, global_step=episode)
                    if episode%10==0:
                        logger.info(f'[{episode}/{n_ep}] <{step}> ep_r:{ep_r/step_inter}, len_mem:{len(self.mem)}, eps:{self.agent_list[0].eps}')
                        #logger.info(f'ep_r_all:{ep_r_all/step_inter}')
                    break

                state_all = torch.tensor(next_state_all, device=device)
            self.mem.commit()

            if len(self.mem) >= self.args.batch_size * 5:
                trans = self.mem.sample(self.args.batch_size)
                loss = self.train_step(*trans)
                loss_sum += loss

                self.writer.add_scalar("loss", loss, global_step=step)
                if episode % self.args.snap == 0:
                    logger.info(f'[{episode}/{n_ep}] <{step}> loss:{loss_sum / self.args.snap}, eps:{self.agent_list[0].eps}')
                    loss_sum = 0

    @logger.catch
    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        self.train()
        self.writer.close()


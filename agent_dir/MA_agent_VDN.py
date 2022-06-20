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

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, n_agent):
        super(QNetwork, self).__init__()
        self.output_size = action_size

        self.base = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
        )

        self.lstm=nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True)

        self.lstm_l = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
        )

        self.head_V=nn.Sequential(
            nn.Linear(512*n_agent, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 1),
        )
        self.head_Adv=nn.Sequential(
            nn.Linear(512*n_agent, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, action_size),
        )

    def step1(self, x):
        x=self.base(x)
        x,(h,c)=self.lstm(x)
        x=self.lstm_l(x)
        return x

    def step2(self, x_all):
        advantage = self.head_Adv(x_all)
        value = self.head_V(x_all)
        return value + advantage - advantage.mean()

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


class AgentVDN():
    def __init__(self, n_act, n_state, n_agent, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        self.n_act = n_act
        self.n_state = n_state
        self.n_agent = n_agent

        self.Qnet = QNetwork(self.n_state, self.n_act, self.n_agent).to(device)

        self.Qnet_T = deepcopy(self.Qnet).to(device)
        for m in self.Qnet_T.parameters():
            m.requires_grad = False

        self.eps = args.eps_start

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.Qnet.parameters(), lr=args.lr, weight_decay=5e-4)

        self.args = args

        self.eps_scd = EpsScheduler(args.eps_start, args.eps_end, args.eps_decay)
        self.ema = EMA(args.ema)

    def soft_update(self):
        self.ema.update_model_average(self.Qnet_T, self.Qnet)

    def eps_update(self):
        self.eps = self.eps_scd.step()

    '''def train_step_pre(self, state, action, reward, next_state, done): #[B,step_ep,N]
        y = deepcopy(reward.float())

        with torch.no_grad():
            not_done = ~done
            y[not_done] += self.args.gamma * self.Qnet_T(next_state[not_done, ...]).max(dim=-1)[0]

        pred = self.Qnet(state, action).view(-1)
        loss = self.criterion(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.Qnet.parameters(), max_norm=self.args.grad_norm_clip)
        self.optimizer.step()

        return loss.item()'''

    def train_step_pre(self, state, next_state):  # [B,step_ep,N]

        with torch.no_grad():
            out_T = self.Qnet_T.step1(next_state)

        out = self.Qnet.step1(state)
        return out, out_T

    def train_step_after(self, out, out_T, action, reward):
        y = deepcopy(reward.float())
        with torch.no_grad():
            y += self.args.gamma * self.Qnet_T.step2(out_T).max(dim=-1)[0]

        pred = self.Qnet.step2(out).gather(1, action).view(-1)
        loss = self.criterion(pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.Qnet.parameters(), max_norm=self.args.grad_norm_clip)
        self.optimizer.step()

        return loss.item()

    #@torch.no_grad()
    def make_action(self, out_all, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return:action
        """
        if test:
            return self.Qnet.step2(out_all).argmax(dim=-1)
        else:
            return self.Qnet.step2(out_all).argmax(dim=-1) if random.random() > self.eps else torch.randint(0, self.n_act, (1,))

class MA_VDN():
    def __init__(self, env, args):
        self.env=env
        self.args=args
        self.writer = SummaryWriter("log")

        self.n_agent = len(env.action_space)
        self.n_act = env.action_space[0].n
        self.n_state = env.observation_space[0].shape[0]

        self.agent_list=[AgentVDN(self.n_act, self.n_state, self.n_agent, args)]*self.n_agent

        self.mem = ReplayBuffer(args.buffer_size)

    def train_step(self, state_all, action_all, reward_all, next_state_all, done_all): # [B,step_ep,n_agent,N]
        self.agent_list[0].soft_update()

        reward_all = (reward_all+7)/10

        loss=0
        out_all=[]
        out_T_all=[]
        for i, agent in enumerate(self.agent_list):
            out, out_T = agent.train_step_pre(state_all[:,:,i,:], next_state_all[:,:,i,:])
            out_all.append(out)
            out_T_all.append(out_T)

        out_all=torch.cat(out_all, dim=-1)
        out_T_all=torch.cat(out_T_all, dim=-1)

        for i, agent in enumerate(self.agent_list):
            loss_i = agent.train_step_after(out_all, out_T_all, action_all[:,:,i], reward_all[:,:,i])
            loss+=loss_i
        return loss/self.n_agent

    @torch.no_grad()
    def make_action_all(self, state_all):
        out_all=[agent.Qnet.step1(state_all[:,i,:].unsqueeze(0)) for i,agent in enumerate(self.agent_list)]
        out_all = torch.cat(out_all, dim=-1)
        return [agent.make_action(out_all).view(-1)[-1] for i,agent in enumerate(self.agent_list)]

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

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        self.train()
        self.writer.close()


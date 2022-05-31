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

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorNetwork, self).__init__()
        self.output_size=action_size

        self.net = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.SiLU(),

            nn.Linear(512, 512),
            nn.SiLU(),

            nn.Linear(512, action_size),
            nn.Tanh()
        )

    def forward(self, inputs):
        return self.net(inputs)

class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(CriticNetwork, self).__init__()
        #self.net = resnet.resnet18(pretrained=True)
        #self.net.conv1 = nn.Conv2d(4, self.net.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        #self.net.fc = nn.Linear(self.net.fc.in_features, output_size)

        #self.net = alexnet(pretrained=True)
        #self.net.features[0]=nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
        #self.net.classifier[-1]=nn.Linear(4096, output_size)

        self.base_state = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.SiLU(),
        )

        self.base_act = nn.Sequential(
            nn.Linear(action_size, 512),
            nn.SiLU(),
        )

        self.net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.SiLU(),

            nn.Linear(512, 256),
            nn.SiLU(),

            nn.Linear(256, 1)
        )

    def forward(self, s, a):

        xa=self.base_act(a)
        xs=self.base_state(s)
        x=torch.cat((xa,xs), dim=1)

        return self.net(x)

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def __len__(self):
        return len(self.buffer)

    def proc(self, state, action, reward, next_state, done):
        return state.float(), action.float(), reward, next_state.float(), done

    def push(self, *transition):
        if len(self.buffer)>=self.buffer_size:
            #self.buffer[random.randint(0,self.buffer_size-1)]=self.proc(*transition)
            self.buffer.pop(0)
            self.buffer.append(self.proc(*transition))
        else:
            self.buffer.append(self.proc(*transition))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return [torch.stack(x, dim=0).to(device) for x in list(zip(*batch))]

    def clean(self):
        self.buffer.clear()


class AgentDDPG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentDDPG, self).__init__(env)

        self.n_act = env.action_space.shape[0]
        self.n_state = env.observation_space.shape[0]

        self.Anet = ActorNetwork(self.n_state, self.n_act).to(device)
        self.Cnet = CriticNetwork(self.n_state, self.n_act).to(device)

        self.Anet_T = deepcopy(self.Anet).to(device)
        self.Cnet_T = deepcopy(self.Cnet).to(device)
        for m in self.Anet_T.parameters():
            m.requires_grad=False
        for m in self.Anet_T.parameters():
            m.requires_grad=False

        self.mem = ReplayBuffer(args.buffer_size)
        self.eps = args.eps_start

        self.criterion = nn.MSELoss()
        self.optimizer_A = torch.optim.Adam(self.Anet.parameters(), lr=args.lr)
        self.optimizer_C = torch.optim.Adam(self.Cnet.parameters(), lr=args.lr)

        self.args=args
        self.writer = SummaryWriter("log")

        self.eps_scd = EpsScheduler(args.eps_start, args.eps_end, args.eps_decay)

    
    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def train_step(self, state, action, reward, next_state, done):
        y = deepcopy(reward.float())
        action = action.view(-1,self.n_act)

        #Actor
        pred = self.Cnet(state, self.Anet(state)).view(-1)
        A_loss = -torch.mean(pred)
        self.optimizer_A.zero_grad()
        A_loss.backward()
        self.optimizer_A.step()

        #Critic
        with torch.no_grad():
            not_done = ~done
            acts=self.Anet_T(next_state[not_done, ...])
            y[not_done] += self.args.gamma*self.Cnet_T(next_state[not_done, ...], acts).view(-1)

        pred = self.Cnet(state, action).view(-1)

        loss = self.criterion(pred, y)
        self.optimizer_C.zero_grad()
        loss.backward()
        self.optimizer_C.step()

        return loss.item(), A_loss.item()

    def train(self):
        """
        Implement your training algorithm here
        """
        n_ep=10000
        step = 0
        loss_sum = 0

        for episode in range(n_ep):
            ep_r = 0

            state = self.env.reset()
            state = torch.tensor(state, device=device)

            while True:
                if self.args.render:
                    self.env.render()

                action = self.make_action(state.unsqueeze(0).float(), self.args.test).detach().cpu()
                self.eps = self.eps_scd.step()

                next_state, reward, done, info = self.env.step(action.view(-1).numpy())

                ep_r += reward

                self.mem.push(*[torch.tensor(x, device='cpu') for x in [state, action, reward, next_state, done]])

                if len(self.mem) >= self.args.batch_size*5:
                    if (step + 1) % self.args.target_update_freq == 0:
                        self.Anet_T.load_state_dict(self.Anet.state_dict())
                        self.Cnet_T.load_state_dict(self.Cnet.state_dict())

                    trans = self.mem.sample(self.args.batch_size)
                    loss, A_loss = self.train_step(*trans)

                    loss_sum += loss
                    if step%self.args.snap==0:
                        self.writer.add_scalar("loss", loss, global_step=step)
                        self.writer.add_scalar("A_loss", A_loss, global_step=step)
                        logger.info(f'[{episode}/{n_ep}] <{step}> loss:{loss_sum/self.args.snap}, eps:{self.eps}')
                        loss_sum = 0
                step += 1

                if done:# or step>self.args.n_frames:
                    self.writer.add_scalar("ep_r", ep_r, global_step=episode)
                    logger.info(f'[{episode}/{n_ep}] <{step}> ep_r:{ep_r}, len_mem:{len(self.mem)}, eps:{self.eps}')
                    break

                state = torch.tensor(next_state, device=device)

                #if (step + 1) % self.args.snap_save == 0:
                #    torch.save({self.Anet.state_dict()}, os.path.join(self.args.save_dir, self.args.name, f'net_{step + 1}.pth'))

    @torch.no_grad()
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return:action
        """
        if test:
            return self.Anet(observation).view(-1)
        else:
            return self.Anet(observation).view(-1) if random.random()>self.eps else torch.rand((self.n_act,))*2-1

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        self.train()
        self.writer.close()
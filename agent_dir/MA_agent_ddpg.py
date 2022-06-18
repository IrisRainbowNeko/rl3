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
        self.output_size = action_size

        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),

            nn.Linear(256, action_size),
            nn.Softmax()
        )

    def forward(self, inputs):
        return self.net(inputs)


class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(CriticNetwork, self).__init__()
        # self.net = resnet.resnet18(pretrained=True)
        # self.net.conv1 = nn.Conv2d(4, self.net.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        # self.net.fc = nn.Linear(self.net.fc.in_features, output_size)

        # self.net = alexnet(pretrained=True)
        # self.net.features[0]=nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
        # self.net.classifier[-1]=nn.Linear(4096, output_size)

        self.base_state = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
        )

        self.base_act = nn.Sequential(
            nn.Linear(action_size, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
        )

        # self.attention = DotProductAttention(512)

        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),

            nn.Linear(256, 1)
        )

    def forward(self, s, a):
        xa = self.base_act(a)
        xs = self.base_state(s)
        x = torch.cat((xa, xs), dim=1)
        # x=self.attention(xs,xa,xs)

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
        if len(self.buffer) >= self.buffer_size:
            # self.buffer[random.randint(0,self.buffer_size-1)]=self.proc(*transition)
            self.buffer.pop(0)
            self.buffer.append(self.proc(*transition))
        else:
            self.buffer.append(self.proc(*transition))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return [torch.stack(x, dim=0).to(device) for x in list(zip(*batch))]

    def clean(self):
        self.buffer.clear()


class AgentDDPG():
    def __init__(self, n_act, n_state, n_agent, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        self.n_act = n_act
        self.n_state = n_state
        self.n_agent = n_agent
        #self.agent_id = agent_id

        self.Anet = ActorNetwork(self.n_state, self.n_act).to(device)
        self.Cnet = CriticNetwork(self.n_state*self.n_agent, self.n_act*self.n_agent).to(device)

        self.Anet_T = deepcopy(self.Anet).to(device)
        self.Cnet_T = deepcopy(self.Cnet).to(device)
        for m in self.Anet_T.parameters():
            m.requires_grad = False
        for m in self.Anet_T.parameters():
            m.requires_grad = False

        self.eps = args.eps_start

        self.criterion = nn.MSELoss()
        #self.criterion = lambda x,y:torch.mean(torch.log(torch.cosh(2*(x-y))))
        self.optimizer_A = torch.optim.AdamW(self.Anet.parameters(), lr=args.lr_a, weight_decay=5e-4)
        self.optimizer_C = torch.optim.AdamW(self.Cnet.parameters(), lr=args.lr_c, weight_decay=5e-4)

        self.args = args

        self.eps_scd = EpsScheduler(args.eps_start, args.eps_end, args.eps_decay)
        self.ema = EMA(args.ema)

    def train_step(self, state_all, action_all, action_all_T, reward, next_state_all, done, agent_id, update=False):
        if update:
            self.ema.update_model_average(self.Anet_T, self.Anet)
            self.ema.update_model_average(self.Cnet_T, self.Cnet)

        y = deepcopy(reward.float())

        # Critic
        with torch.no_grad():
            #not_done = ~done
            #acts = action_all_T[not_done, ...].flatten(1)
            #y[not_done] += self.args.gamma * self.Cnet_T(next_state_all[not_done, ...].flatten(1), acts).view(-1)
            acts = action_all_T.flatten(1)
            y += self.args.gamma * self.Cnet_T(next_state_all.flatten(1), acts).view(-1)

        pred = self.Cnet(state_all.flatten(1), action_all.flatten(1)).view(-1)
        loss = self.criterion(pred, y)

        self.optimizer_C.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.Cnet.parameters(), max_norm=self.args.grad_norm_clip, norm_type=2)
        self.optimizer_C.step()

        # Actor
        A_loss=[]
        for i in range(self.n_agent):
            action_all_i = deepcopy(action_all)
            action_all_i[:,i,:]=self.Anet(state_all[:,i,:])
            pred = self.Cnet(state_all.flatten(1), action_all_i.flatten(1)).view(-1)
            A_loss.append(-torch.mean(pred))
        A_loss=sum(A_loss)/self.n_agent

        self.optimizer_A.zero_grad()
        A_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.Anet.parameters(), max_norm=self.args.grad_norm_clip, norm_type=2)
        self.optimizer_A.step()

        return loss.item(), A_loss.item()

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
            if random.random() > self.eps:
                return self.Anet(observation).view(-1)
            else:
                act = torch.zeros((self.n_act,))
                act[random.randint(0, self.n_act - 1)] = 1
                return act

class MA_DDPG():
    def __init__(self, env, args):
        self.env=env
        self.args=args
        self.writer = SummaryWriter("log")

        self.n_agent = len(env.action_space)
        self.n_act = env.action_space[0].n
        self.n_state = env.observation_space[0].shape[0]

        #self.agent_list=[AgentDDPG(self.n_act, self.n_state, self.n_agent, i, args) for i in range(self.n_agent)]
        self.agent_list=[AgentDDPG(self.n_act, self.n_state, self.n_agent, args)]*self.n_agent

        self.mem = ReplayBuffer(args.buffer_size)

    def train_step(self, state_all, action_all, reward_all, next_state_all, done_all):
        with torch.no_grad():
            action_all_T = torch.stack([agent.Anet_T(next_state_all[:,i,:]) for i,agent in enumerate(self.agent_list)], dim=1)

        reward_all = (reward_all+7)/10

        C_loss=0
        A_loss=0
        for i, agent in enumerate(self.agent_list[0:1]):
            C_loss_i, A_loss_i = agent.train_step(state_all, action_all, action_all_T, reward_all[:,i], next_state_all,
                                                  done_all[:,i], i, update=(i==0))
            C_loss+=C_loss_i
            A_loss+=A_loss_i
        return C_loss/self.n_agent, A_loss/self.n_agent

    def train(self):
        """
        Implement your training algorithm here
        """
        n_ep = self.args.n_ep
        step = 0
        loss_sum_C = 0
        loss_sum_A = 0

        for episode in range(n_ep):
            ep_r = 0
            ep_r_all = np.zeros(3)
            step_inter=0

            state_all = self.env.reset()
            state_all = torch.tensor(state_all, device=device)

            while True:
                if self.args.render:
                    self.env.render(mode=None)

                action_list = [agent.make_action(state_all[i].unsqueeze(0).float(), self.args.test).detach().cpu() for i,agent in enumerate(self.agent_list)]
                action_all = torch.stack(action_list, dim=0)

                for agent in self.agent_list:
                    agent.eps=agent.eps_scd.step()

                next_state_list, reward_list, done_list, info = self.env.step([to_onehot(x.view(-1)).numpy() for x in action_list])
                next_state_all=torch.tensor(next_state_list)
                reward_all=torch.tensor(reward_list)
                done_all=torch.tensor(done_list)

                ep_r += reward_all.mean()
                ep_r_all += reward_all.numpy()

                self.mem.push(*[torch.tensor(x, device='cpu') for x in [state_all.cpu(), action_all, reward_all, next_state_all, done_all]])

                if len(self.mem) >= self.args.batch_size * 5:
                    trans = self.mem.sample(self.args.batch_size)
                    C_loss, A_loss = self.train_step(*trans)

                    loss_sum_C += C_loss
                    loss_sum_A += A_loss
                    if step % self.args.snap == 0:
                        self.writer.add_scalar("loss", C_loss, global_step=step)
                        self.writer.add_scalar("A_loss", A_loss, global_step=step)
                        logger.info(f'[{episode}/{n_ep}] <{step}> loss_C:{loss_sum_C / self.args.snap}, loss_A:{loss_sum_A / self.args.snap}, eps:{self.agent_list[0].eps}')
                        loss_sum_C = 0
                        loss_sum_A = 0
                step += 1
                step_inter += 1

                if done_all.all() or step_inter>self.args.max_step:
                    self.writer.add_scalar("ep_r", ep_r/step_inter, global_step=episode)
                    if episode%10==0:
                        logger.info(f'[{episode}/{n_ep}] <{step}> ep_r:{ep_r/step_inter}, len_mem:{len(self.mem)}, eps:{self.agent_list[0].eps}')
                        #logger.info(f'ep_r_all:{ep_r_all/step_inter}')
                    break

                state_all = torch.tensor(next_state_all, device=device)

                # if (step + 1) % self.args.snap_save == 0:
                #    torch.save({self.Anet.state_dict()}, os.path.join(self.args.save_dir, self.args.name, f'net_{step + 1}.pth'))

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        self.train()
        self.writer.close()


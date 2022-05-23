import os
import random
from copy import deepcopy
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
from agent_dir.agent import Agent
from loguru import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PGNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PGNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.SiLU(),

            nn.Linear(512, 512),
            nn.SiLU(),

            nn.Linear(512, output_size)
        )

    def forward(self, inputs):
        return self.net(inputs)


class ReplayBuffer:
    def __init__(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def proc(self, state, action, reward):
        return state.float(), action, reward

    def push(self, *transition):
        self.buffer.append(self.proc(*transition))

    def sample(self):
        return [torch.stack(x, dim=0).to(device) for x in list(zip(*self.buffer))]

    def clean(self):
        self.buffer.clear()


class AgentPG(Agent):
    def __init__(self, env, args, network=PGNetwork):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentPG, self).__init__(env)

        self.n_act = env.action_space.n

        self.Qnet = network(4, self.n_act).to(device)
        self.Qnet_T = deepcopy(self.Qnet).to(device)
        for m in self.Qnet_T.parameters():
            m.requires_grad=False

        self.mem = ReplayBuffer()

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.Qnet.parameters(), lr=args.lr)

        self.args=args
        self.writer = SummaryWriter("log")

        #self.eps_scd = EpsScheduler(args.eps_start, args.eps_end, args.eps_decay)

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        pass

    def train_step(self, state, action, reward):
        reward_dc = torch.empty_like(reward, device=device)
        running_add=0
        for t in reversed(range(0, len(reward))):  # 反向计算
            running_add = running_add * self.args.gamma + reward[t]
            reward_dc[t] = running_add

        reward_dc -= reward_dc.mean()
        reward_dc /= reward_dc.std()

        pred = self.Qnet(state)

        loss = self.criterion(pred, action)
        loss = torch.mean(loss * reward_dc)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.mem.clean()

        return loss.item()

    def train(self):
        """
        Implement your training algorithm here
        """

        n_ep = 10000
        step = 0
        loss_sum = 0

        for episode in range(n_ep):
            ep_r = 0

            state = self.env.reset()
            state = torch.tensor(state, device=device)

            while True:
                if self.args.render:
                    self.env.render()

                action = self.make_action(state.unsqueeze(0).float(), self.args.test)
                next_state, reward, done, info = self.env.step(action.item())

                ep_r += reward

                self.mem.push(*[torch.tensor(x, device='cpu') for x in [state, action, reward]])

                state = torch.tensor(next_state, device=device)
                step += 1

                if done:  # or step>self.args.n_frames:
                    self.writer.add_scalar("ep_r", ep_r, global_step=episode)

                    trans = self.mem.sample()
                    loss = self.train_step(*trans)
                    logger.info(f'[{episode}/{n_ep}] <{step}> ep_r:{ep_r}, loss:{loss}')

                    if (episode + 1) % self.args.snap_save == 0:
                        torch.save(self.Qnet.state_dict(), os.path.join(self.args.save_dir, self.args.name, f'net_{step + 1}.pth'))
                    break



    @torch.no_grad()
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return: action
        """
        act = torch.softmax(self.Qnet(observation).view(-1), dim=0).cpu().numpy()
        return torch.tensor(np.random.choice(range(act.shape[0]), p=act))

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        self.train()
        self.writer.close()

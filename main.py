import argparse
import os

from wrappers import make_env, ProcessFrame84
import gym
from argument import dqn_arguments, pg_arguments
import torch
import cv2

def parse():
    parser = argparse.ArgumentParser(description="SYSU_RL_HW2")
    parser.add_argument('--train_pg', default=False, type=bool, help='whether train policy gradient')
    parser.add_argument('--train_dqn', default=True, type=bool, help='whether train DQN')

    parser.add_argument("--render", default=False, type=bool)
    parser.add_argument("--snap", default=100, type=float)
    parser.add_argument("--name", default='CartPole', type=str)
    parser.add_argument("--snap_save", default=10000, type=float)
    parser.add_argument("--save_dir", default='output', type=str)

    parser = dqn_arguments(parser)
    #parser = pg_arguments(parser)
    args = parser.parse_args()
    return args


def run(args):
    if args.train_pg:
        env_name = args.env_name
        env = gym.make(env_name)
        from agent_dir.agent_pg import AgentPG
        agent = AgentPG(env, args)
        agent.run()

    if args.train_dqn:
        env_name = args.env_name
        env = make_env(env_name)
        #env = gym.make(env_name)
        from agent_dir.agent_dqn import AgentDQN, AgentDDQN, AgentDuelingDQN

        agent = eval(args.agent)(env, args)
        agent.run()

def test(args):
    env_name = args.env_name
    env = make_env(env_name)
    state=env.reset()

    print(env.action_space.n)
    print(state.shape)
    #print(env.observation_space)

    state = env.step(1)[0]
    for i in range(60):
        state = env.step(2)[0]

    #state=torch.tensor(state).permute(2,0,1)

    from matplotlib import pyplot as plt
    plt.figure()
    for i in range(3):
        plt.subplot(2,2,1+i)
        plt.imshow(state[i, :,:])
    plt.show()


if __name__ == '__main__':
    args = parse()
    args.save_dir = f'{args.save_dir}_{args.agent}'
    os.makedirs(os.path.join(args.save_dir, args.name), exist_ok=True)
    run(args)

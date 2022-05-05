import argparse
from wrappers import make_env, ProcessFrame84
import gym
from argument import dqn_arguments, pg_arguments
import torch

def parse():
    parser = argparse.ArgumentParser(description="SYSU_RL_HW2")
    parser.add_argument('--train_pg', default=False, type=bool, help='whether train policy gradient')
    parser.add_argument('--train_dqn', default=True, type=bool, help='whether train DQN')


    parser.add_argument("--snap", default=100, type=float)

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
        from agent_dir.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
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
    for i in range(4):
        plt.subplot(2,2,1+i)
        plt.imshow(state[i, :,:])
    plt.show()


if __name__ == '__main__':
    args = parse()
    run(args)

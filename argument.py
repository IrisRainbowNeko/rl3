def str2list(v):
    return [float(x) for x in v.split(',')]

# python main.py --render True --batch_size 256 --lr_a 0.0008 --eps_decay 100000 --ema 0.99
def maddpg_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    #parser.add_argument('--env_name', default="LunarLanderContinuous-v2", help='environment name')
    parser.add_argument('--env_name', default="simple_spread.py", help='environment name')

    parser.add_argument('--agent', default="MA_DDPG", help='agent name')

    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--buffer_size", default=int(1e6), type=int)
    parser.add_argument("--lr_c", default=0.001, type=float)
    parser.add_argument("--lr_a", default=0.0008, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--grad_norm_clip", default=1, type=float)

    parser.add_argument("--max_step", default=25, type=int)
    parser.add_argument("--n_ep", default=50000, type=int)

    parser.add_argument("--eps_start", default=1.0, type=float)
    parser.add_argument("--eps_end", default=0.01, type=float)
    parser.add_argument("--eps_decay", default=100000, type=float)

    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--n_frames", default=int(30000), type=int)
    parser.add_argument("--ema", default=0.99, type=float)

    return parser

def VDN_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    #parser.add_argument('--env_name', default="LunarLanderContinuous-v2", help='environment name')
    parser.add_argument('--env_name', default="simple_spread.py", help='environment name')

    parser.add_argument('--agent', default="MA_VDN", help='agent name')

    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--buffer_size", default=int(1e6), type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--grad_norm_clip", default=1, type=float)

    parser.add_argument("--max_step", default=25, type=int)
    parser.add_argument("--n_ep", default=50000, type=int)

    parser.add_argument("--eps_start", default=1.0, type=float)
    parser.add_argument("--eps_end", default=0.01, type=float)
    parser.add_argument("--eps_decay", default=500000, type=float)

    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--n_frames", default=int(30000), type=int)
    parser.add_argument("--ema", default=0.99, type=float)

    return parser

def QMIX_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    #parser.add_argument('--env_name', default="LunarLanderContinuous-v2", help='environment name')
    parser.add_argument('--env_name', default="simple_spread.py", help='environment name')

    parser.add_argument('--agent', default="MA_QMIX", help='agent name')

    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--buffer_size", default=int(1e6), type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--grad_norm_clip", default=1, type=float)

    parser.add_argument("--max_step", default=25, type=int)
    parser.add_argument("--n_ep", default=50000, type=int)

    parser.add_argument("--eps_start", default=1.0, type=float)
    parser.add_argument("--eps_end", default=0.01, type=float)
    parser.add_argument("--eps_decay", default=500000, type=float)

    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--n_frames", default=int(30000), type=int)
    parser.add_argument("--ema", default=0.99, type=float)

    parser.add_argument("--sig", default=False, type=bool)

    return parser
def str2list(v):
    return [float(x) for x in v.split(',')]

def dqn_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="PongNoFrameskip-v4", help='environment name')
    #parser.add_argument('--env_name', default="CartPole-v0", help='environment name')

    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--buffer_size", default=int(3e4), type=int)
    parser.add_argument("--mem_step", default=int(5e3), type=int)
    parser.add_argument("--lr", default=0.0002, type=float)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--eps", default=0.9, type=float)
    parser.add_argument("--eps_decay", default=[87000, 93000, 96000, 98000, 99000, 100000, 110000], type=str2list)
    parser.add_argument("--eps_decay_rate", default=0.8, type=str2list)
    parser.add_argument("--grad_norm_clip", default=10, type=float)

    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--n_frames", default=int(400000), type=int)
    parser.add_argument("--learning_freq", default=1, type=int)
    parser.add_argument("--target_update_freq", default=5000, type=int)

    return parser

def pg_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="PongNoFrameskip-v4", help='environment name')

    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--hidden_size", default=16, type=int)
    parser.add_argument("--lr", default=0.02, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--grad_norm_clip", default=10, type=float)

    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--n_frames", default=int(30000), type=int)

    return parser

import gym
import mjrl.envs
#import trajopt.envs
import mj_envs
import click
import os
import gym
import numpy as np
import pickle
import torch
from mjrl.utils.gym_env_rnn import GymEnv
from mjrl.policies.gaussian_mlp import MLP
import torch
from torch.autograd import Variable

DESC = '''
Helper script to visualize policy (in mjrl format).\n
USAGE:\n
    Visualizes policy on the env\n
    $ python utils/visualize_policy --env_name mjrl_swimmer-v0 --policy my_policy.pickle --mode evaluation --episodes 10 \n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required= True)
@click.option('--policy', type=str, help='absolute path of the policy file', default=None)
@click.option('--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
@click.option('--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('--episodes', type=int, help='number of episodes to visualize', default=10)
@click.option('--log_std', type=float, default=-0.5)
@click.option('--terminate', type=bool, default=True)
@click.option('--device_path', type=str, default=None)
def main(env_name, policy, mode, seed, episodes, log_std, terminate, device_path):
    with open("utils/bcrl.txt", 'r') as f:
        job_data = eval(f.read())
    demo_paths = pickle.load(open('demonstrations/relocate-v0_demos.pickle', 'rb'))
    train_data = demo_paths[:-5]
    #train_data = demo_paths[-5:]


    render = True

    # TODO(Aravind): Map to hardware if device_path is specified
    seed = 123
    e = GymEnv(env_name)
    e.set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if policy is not None:
        policy = pickle.load(open(policy, 'rb'))
    else:
        raise Exception
        policy = MLP(e.spec, hidden_sizes=(32,32), seed=seed, init_log_std=log_std)
    
    for ep in range(len(train_data)):
        train_ep = train_data[ep]
        train_init = train_ep['observations'][0]
        o = e.reset()
        e.set_env_state(train_ep['init_state_dict'])
        rew = 0.0
        t = 0
        done = False
        while t < e.horizon and done is False:
            o = e.get_obs() if t > 0 else train_init
            if type(o) is not torch.Tensor:
                ob = Variable(torch.from_numpy(o).float(), requires_grad=False)
            ob = torch.unsqueeze(ob, 0)
            ob = torch.unsqueeze(ob, 0)
            obs = torch.cat((obs, ob),1) if t > 0 else ob
            a = policy.get_action(obs.detach())[0] #if mode == 'exploration' else policy.get_action(obs)[1]['evaluation']
            if type(a) is not torch.Tensor:
                ac = Variable(torch.from_numpy(a).float(), requires_grad=False)
            ac = torch.unsqueeze(ac, 0)
            ac = torch.unsqueeze(ac, 0)
            acts = torch.cat((acts, ac),1) if t > 0 else ac
            next_o, r, done, ifo = e.step(a)
            if terminate is False:
                done = False
            rew = rew + r
            t = t + 1
            if render:
                e.render()
            if done and t < e.horizon - 1:
                print("Episode terminated early")
        print("episode score = %f " % rew)

    e.reset()


if __name__ == '__main__':
    main()

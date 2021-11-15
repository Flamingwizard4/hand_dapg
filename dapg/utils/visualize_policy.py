import mj_envs
import click 
import os
import gym
import numpy as np
import pickle
from mjrl.utils.gym_env import GymEnv

DESC = '''
Helper script to visualize policy (in mjrl format).\n
USAGE:\n
    Visualizes policy on the env\n
    $ python utils/visualize_policy --env_name relocate-v0 --policy policies/relocate-v0.pickle --mode evaluation\n
'''

# MAIN =========================================================
@click.command(help=DESC)
@click.option('--env_name', type=str, help='environment to load', required= True)
@click.option('--policy', type=str, help='absolute path of the policy file', required=True)
@click.option('--mode', type=str, help='exploration or evaluation mode for policy', default='evaluation')
def main(env_name, policy, mode):
    with open("utils/bcrl.txt", 'r') as f:
        job_data = eval(f.read())
    demo_paths = pickle.load(open('demonstrations/relocate-v0_demos.pickle', 'rb'))
    #train_data = demo_paths[:-5]
    train_data = demo_paths[-5:]

    e = GymEnv(env_name)
    pi = pickle.load(open(policy, 'rb'))
    # render policy
    #e.visualize_policy(pi, num_episodes=100, horizon=e.horizon, mode=mode)
    for ep in range(len(train_data)):
        train_ep = train_data[ep]
        train_init = train_ep['observations'][0]
        o = e.reset()
        d = False
        t = 0
        score = 0.0
        while t < e.horizon and d is False:
            if t == 0:
                o = train_init
            a = pi.get_action(o)[0] if mode == 'exploration' else pi.get_action(o)[1]['evaluation']
            o, r, d, _ = e.step(a)
            score = score + r
            e.render()
            t = t+1
        print("Episode score = %f" % score)

if __name__ == '__main__':
    main()

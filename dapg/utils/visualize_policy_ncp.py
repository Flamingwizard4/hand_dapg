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
from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.policies.ncp_seq import NCP_sequence
from mjrl.utils.ncp_model import SequenceLearner
import torch
from torch.autograd import Variable
import kerasncp as kncp
from kerasncp.torch import LTCCell
#import pytorch_lightning as pl

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
    # demo_playback(env_name, demo_paths)
    test_data = demo_paths[-1:]
    #train_data = demo_paths[-5:]
    # print("Observations: ")
    # for i in range(len(test_data[0]['observations'])):
    #     if i > len(test_data[0]['observations']) - 5:
    #         print(test_data[0]['observations'][i])
    #         print(p_obs - test_data[0]['observations'][i])
    #     p_obs = test_data[0]['observations'][i]
    # print("Expert Actions: ")
    # for i in range(len(test_data[0]['actions'])):
    #     if i > len(test_data[0]['actions']) - 5:
    #         print(test_data[0]['actions'][i])
    #         print(p_act - test_data[0]['actions'][i])
    #     p_act = test_data[0]['actions'][i]
    # raise Exception
    render = True

    # TODO(Aravind): Map to hardware if device_path is specified
    seed = 123
    e = GymEnv(env_name)
    # e.set_seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    
    ncp_wiring = kncp.wirings.NCP(
        inter_neurons=12,  # Number of inter neurons
        command_neurons=6,  # Number of command neurons
        motor_neurons=e.spec.action_dim,  # Number of motor neurons
        sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
        inter_fanout=6,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=6,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=5,  # How many incomming syanpses has each motor neuron
    )
    # ncp_wiring = kncp.wirings.NCP(
    #     inter_neurons=6,  # Number of inter neurons
    #     command_neurons=8,  # Number of command neurons
    #     motor_neurons=e.spec.action_dim,  # Number of motor neurons
    #     sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
    #     inter_fanout=4,  # How many outgoing synapses has each inter neuron
    #     recurrent_command_synapses=6,  # Now many recurrent synapses are in the
    #     # command neuron layer
    #     motor_fanin=8,  # How many incomming syanpses has each motor neuron
    # )
    ltc_cells = LTCCell(ncp_wiring, e.spec.observation_dim)
    ncp_sequence = NCP_sequence(ltc_cells)
    learn = SequenceLearner(ncp_sequence)
    # for dir in os.listdir("examples/log/default/version_1/checkpoints"):
    #     ncp_model = learn.load_from_checkpoint(os.path.join("examples/log/default/version_3/checkpoints", dir), model=learn.model)
    for dir in os.listdir("examples/tb_logs/ncp_model/version_1/checkpoints"):
        ncp_model = learn.load_from_checkpoint(os.path.join("examples/tb_logs/ncp_model/version_1/checkpoints", dir), model=learn.model)
    # if policy is not None:
    #     policy = pickle.load(open(policy, 'rb'))
    # else:
    #     raise Exception
    #     policy = MLP(e.spec, hidden_sizes=(32,32), seed=seed, init_log_std=log_std)
    
    # train_observations = [path["observations"] for path in demo_paths[:20]]
    # test_observations = [path["observations"] for path in demo_paths[20:]]
    test_observations = [path["observations"] for path in demo_paths[-1:]]
    # train_s = [path["init_state_dict"] for path in demo_paths[:20]]
    test_s = [path["init_state_dict"] for path in demo_paths[-1:]]
    # max_obs_len = len(max(train_observations, key=len))
    # train_expert_action_paths = [path["actions"] for path in demo_paths[:20]]
    # test_expert_action_paths = [path["actions"] for path in demo_paths[20:]]
    test_expert_action_paths = [path["actions"] for path in demo_paths[-1:]]
    # max_act_len = len(max(train_expert_action_paths, key=len))
    # assert max_obs_len == max_act_len
    # train_obs, train_act = pad_paths(train_observations, train_expert_action_paths, max_act_len)
    # test_obs, test_act = pad_paths(test_observations, test_expert_action_paths, max_act_len)
    test_obs, test_act = test_observations, test_expert_action_paths
    for ep in range(len(test_data)):
        # train_ep = test_data[ep]
        # train_init = train_ep['observations'][0]
        e.reset()
        rew = 0.0
        t = 0
        done = False
        e.set_env_state(test_s[ep])
        while t < e.horizon and done is False:

            o = e.get_obs()
            # o = train_observations[ep][t]
            # o = test_observations[ep][t]

            if type(o) is not torch.Tensor:
                ob = Variable(torch.from_numpy(o).float(), requires_grad=False)
            ob = torch.unsqueeze(ob, 0)
            ob = torch.unsqueeze(ob, 0)
            obs = torch.cat((obs, ob),1) if t > 0 else ob
            a = ncp_model.model.forward(ob).detach()
            # print(a)
            # print(test_act[ep][t])
            ac = a.numpy().ravel()
            # if type(ac) is not torch.Tensor:
            #     ac = Variable(torch.from_numpy(ac).float(), requires_grad=False)
            # e_ac = train_act[ep][t]
            # ac = test_act[ep][t]
            #print(((np.array(e_ac) - np.array(a))**2).mean())
            # print(a.shape)
            # ac = torch.unsqueeze(ac, 0)
            # ac = torch.unsqueeze(ac, 0)
            #print(ac.shape)
            acts = torch.cat((acts, a),1) if t > 0 else a
            c_pred_o = test_observations[ep][t]
            pred_o = test_observations[ep][t+1]
            next_o, r, done, ifo = e.step(ac)
            # if t < 200:
                # print("Current Observation:", o)
                # print("Current Predicted Observation:", c_pred_o)
                # print(mse_loss(torch.from_numpy(o), torch.from_numpy(c_pred_o)))
                # print("Predicted Observation:",pred_o)
                # print("Next Observation:",next_o)
                # print(mse_loss(torch.from_numpy(pred_o), torch.from_numpy(next_o)))
            if terminate is False:
                done = False
            rew = rew + r
            t = t + 1
            if render:
                # e.env.mj_render()
                e.render()
            if done and t < e.horizon - 1:
                print("Episode terminated early")
        print("episode score = %f " % rew)

    e.reset()

def pad_paths(obs, act, m):
    new_obs, new_act = [], []
    for i in range(len(obs)):
        o, a = obs[i], act[i]
        n = o.shape[0]
        last_o = np.reshape(o[-1][:], (1, o.shape[1])) #should be outside the loop but too lazy
        #last_a = np.reshape(a[-1][:], (1, a.shape[1]))
        null_a = np.zeros((1, a.shape[1]))
        while n < m:
            o = np.concatenate((o, last_o))
            a = np.concatenate((a, null_a))
            n += 1
        new_obs.append(o)
        new_act.append(a)
    return new_obs, new_act

def demo_playback(env_name, demo_paths):
    e = GymEnv(env_name)
    e.reset()
    for path in demo_paths:
        e.set_env_state(path['init_state_dict'])
        actions = path['actions']
        for t in range(actions.shape[0]):
            e.step(actions[t])
            e.env.mj_render()

if __name__ == '__main__':
    main()


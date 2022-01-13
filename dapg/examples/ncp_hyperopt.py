"""
This is a job script for running NPG/DAPG on hand tasks and other gym envs.
Note that DAPG generalizes PG and BC init + PG finetuning.
With appropriate settings of parameters, we can recover the full family.
"""

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.rnn import RNN
#from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
#from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.algos.dapg import DAPG
from mjrl.algos.rnn_behavior_cloning import rnn_BC
from mjrl.utils.train_agent import train_agent
from mjrl.samplers.core import sample_paths
import os
import pandas as pd
import json
import mjrl.envs
import mj_envs
import time as timer
import pickle
import argparse
from matplotlib import pyplot as plt
import kerasncp as kncp
from kerasncp.torch import LTCCell
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import shutil
from mjrl.policies.ncp_seq import NCP_sequence
from mjrl.utils.ncp_model import SequenceLearner
from pytorch_lightning.loggers import TensorBoardLogger
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL

# devices = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
# d_n = [torch.cuda.get_device_name(d) for d in devices]
# print(d_n)
# print("CUDA Availability: ", torch.cuda.is_available())
# print("Current Device: ", torch.cuda.current_device())
# ===============================================================================
# Get command line arguments
# ===============================================================================

parser = argparse.ArgumentParser(description='Policy gradient algorithms with demonstration data.')
#parser.add_argument('--output', type=str, required=True, help='location to store results')
parser.add_argument('--config', type=str, required=True, help='path to config file with exp params')
args = parser.parse_args()
'''JOB_DIR = args.output
if not os.path.exists(JOB_DIR):
    os.mkdir(JOB_DIR)'''
with open(args.config, 'r') as f:
    job_data = eval(f.read())
assert 'algorithm' in job_data.keys()
assert any([job_data['algorithm'] == a for a in ['NPG', 'BCRL', 'DAPG']])
job_data['lam_0'] = 0.0 if 'lam_0' not in job_data.keys() else job_data['lam_0']
job_data['lam_1'] = 0.0 if 'lam_1' not in job_data.keys() else job_data['lam_1']
EXP_FILE = os.getcwd() + 'job_config.json' #EXP_FILE = JOB_DIR + '/job_config.json'
with open(EXP_FILE, 'w') as f:
    json.dump(job_data, f, indent=4)

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

def augmentation_slicing_total(observations, actions, n):
    new_obs, new_act = [], []
    for i in range(len(observations)):
        if not (len(observations[i]) == len(actions[i])):
            print("unaligned dataset")
            raise Exception
        obs, act = observations[i], actions[i]
        for t in range(len(obs)-(n+1)):
            s_obs = obs[t:t+n]
            s_act = act[t:t+n]
            new_obs.append(s_obs)
            new_act.append(s_act)
    return new_obs, new_act

def objective(params):
    e = GymEnv(job_data['env'])
    avg_train_rewards, avg_test_rewards = [], []

    #policy = RNN(e.spec, hidden_size=job_data['bc_hidden_size'], n_layers=n, seed=job_data['seed']) #job_data['policy_size']
    #policy = MLP(e.spec, hidden_sizes=job_data['policy_size'], seed=job_data['seed'])
    #baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=job_data['vf_batch_size'],
                        #epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'])
    print("========================================")
    print("Collecting expert demonstrations")
    print("========================================")
    demo_paths = pickle.load(open(job_data['demo_file'], 'rb'))
    # bc_agent = rnn_BC(demo_paths, policy=policy, env=e, epochs=job_data['bc_epochs'], seed=job_data['seed'], batch_size=job_data["bc_batch_size"],
    #             lr=job_data['bc_learn_rate'], loss_type='MSE', set_transforms=False)
    # in_shift, in_scale, out_shift, out_scale = bc_agent.compute_transformations()
    # bc_agent.set_transformations(in_shift, in_scale, out_shift, out_scale)
    # bc_agent.set_variance_with_data(out_scale)

    train_observations = [path["observations"] for path in demo_paths[:20]]
    val_observations = [path["observations"] for path in demo_paths[20:-1]]
    test_observations = [path["observations"] for path in demo_paths[-1:]]
    max_obs_len = len(max(train_observations, key=len))
    train_expert_action_paths = [path["actions"] for path in demo_paths[:20]]
    val_expert_action_paths = [path["actions"] for path in demo_paths[20:-1]]
    test_expert_action_paths = [path["actions"] for path in demo_paths[-1:]]
    max_act_len = len(max(train_expert_action_paths, key=len))
    assert max_obs_len == max_act_len
    obs, act = pad_paths(train_observations, train_expert_action_paths, max_act_len)
    v_obs, v_act = pad_paths(val_observations, val_expert_action_paths, max_act_len)
    # obs, act = test_observations, test_expert_action_paths
    # v_obs, v_act = test_observations, test_expert_action_paths
    t_obs, t_act = test_observations, test_expert_action_paths
    # aug_obs, aug_act = augmentation_slicing_total(train_observations, train_expert_action_paths, 42)
    # v_aug_obs, v_aug_act = augmentation_slicing_total(test_observations, test_expert_action_paths, 42)
    # print(len(aug_obs))
    # print(len(aug_act))
    # obs, act, v_obs, v_act = train_observations, train_expert_action_paths, test_observations, test_expert_action_paths
    train_obs = [torch.from_numpy(x) for x in obs]
    train_act = [torch.from_numpy(y) for y in act]
    val_obs = [torch.from_numpy(x) for x in v_obs]
    val_act = [torch.from_numpy(y) for y in v_act]
    test_obs = [torch.from_numpy(x) for x in t_obs]
    test_act = [torch.from_numpy(y) for y in t_act]
    # train_obs = [torch.from_numpy(x) for x in aug_obs]
    # train_act = [torch.from_numpy(y) for y in aug_act]
    # test_obs = [torch.from_numpy(x) for x in v_aug_obs]
    # test_act = [torch.from_numpy(x) for x in v_aug_act]

    # print("========================================")
    # print("Running NCP BC with %d-batch size on expert demonstrations for %d epochs"%(job_data["bc_batch_size"], job_data['bc_epochs']))
    # print("========================================")
    # history_ncp = ncp_model.fit(x=obs, y=act, batch_size=job_data["bc_batch_size"], epochs=job_data['bc_epochs'], verbose=1)

    ###Code below taken from Neural Circuit Policies github pt_example.py
    in_features = e.spec.observation_dim
    out_features = e.spec.action_dim

    data_x = torch.stack(train_obs, dim=0)
    data_y = torch.stack(train_act, dim=0)
    v_data_x = torch.stack(val_obs, dim=0)
    v_data_y = torch.stack(val_act, dim=0)
    t_data_x = torch.stack(test_obs, dim=0)
    t_data_y = torch.stack(test_act, dim=0)
    
    ###NCP Policy
    # wiring = kncp.wirings.FullyConnected(8, out_features)  # 38 units, 30 motor neurons
    ncp_wiring = kncp.wirings.NCP(
        inter_neurons=int(params["i_n"]),  # Number of inter neurons
        command_neurons=int(params["c_n"]),  # Number of command neurons
        motor_neurons=out_features,  # Number of motor neurons
        sensory_fanout=int(params["s_f"]),  # How many outgoing synapses has each sensory neuron
        inter_fanout=int(params["i_f"]),  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=int(params["c_r"]),  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=int(params["m_f"]),  # How many incomming syanpses has each motor neuron
    )
    ltc_cells = LTCCell(ncp_wiring, in_features)

    dataloader = data.DataLoader(
        data.TensorDataset(data_x, data_y), batch_size=16, shuffle=True, num_workers=4
    ) #job_data["bc_batch_size"]
    v_dataloader = data.DataLoader(
        data.TensorDataset(v_data_x, v_data_y), batch_size=4, shuffle=False, num_workers=4
    )
    t_dataloader = data.DataLoader(
        data.TensorDataset(t_data_x, t_data_y), batch_size=1, shuffle=False, num_workers=4
    )

    ltc_sequence = NCP_sequence(
        ltc_cells,
    )
    policy = NCP_sequence
    learn = SequenceLearner(ltc_sequence, lr=0.01)

    # for dir in os.listdir("log/default"):
    #     shutil.rmtree(os.path.join("log/default", dir))

    trainer = pl.Trainer(
        logger=TensorBoardLogger('tb_logs', name='ncp_hyperopt'),#pl.loggers.CSVLogger("log"),
        max_epochs=job_data['bc_epochs'],
        progress_bar_refresh_rate=1,
        gradient_clip_val=1,  # Clip gradient to stabilize training
        gpus=1,
        log_every_n_steps=5
        # limit_train_batches=0,
        # limit_val_batches=0
    )
    # ts = timer.time()
    try:
        val_results = trainer.fit(learn, dataloader, v_dataloader)
    except:
        return {
            'status': STATUS_FAIL
        }
    # print("========================================")
    # print("BC training complete !!!")
    # print("time taken = %f" % (timer.time() - ts))
    # print("========================================")
    results = trainer.test(learn, t_dataloader)
    # print("Validation Results:")
    # print(results)
    # print(type(results[0]['val_loss']))
    return {
        'loss': results[0]['val_loss'],
        'status': STATUS_OK,
        'val_losses': val_results
    }

# ===============================================================================
# Train Loop
# ===============================================================================

trials = Trials()
ncp_param_space = {
    's_f': hp.quniform("s_f", 2, 5, 1),
    'i_n': hp.quniform("i_n", 6, 16, 1),
    'i_f': hp.quniform("i_f", 2, 6, 1),
    'c_n': hp.quniform("c_n", 6, 12, 1),
    'c_r': hp.quniform("c_r", 1, 6, 1),
    'm_f': hp.quniform("m_f", 1, 6, 1)
}
best = fmin(objective,
    space=ncp_param_space,
    algo=tpe.suggest,
    max_evals=200,
    trials=trials)
print(best)
#Hyperopt Bounds:

#best loss: 0.013799767009913921] {'c_n': 6.0, 'c_r': 5.0, 'i_f': 6.0, 'i_n': 12.0, 'm_f': 5.0, 's_f': 4.0} 100 epochs for 100 max evals, bounds above
# if job_data['eval_rollouts'] >= 1:
#     score = e.evaluate_policy(policy, num_episodes=job_data['eval_rollouts'], mean_action=True, seed=321)
#     print("Score with behavior cloning = %f" % score[0][0])
#     print("Performance with BC: %d / %d"%(score[0][4], job_data['eval_rollouts']))
# train_losses, test_losses = e.evaluate_train_vs_test(bc_agent, demo_paths)
# train_score, test_score, p = 0, 0, 0
# for r in train_losses:
#     #print("Training loss with mse cloning = %f" % r[0])
#     #print("Score with selfrolled mse = %f" % r[1])
#     #print("Performance with selfrolled policy: %d" % r[2])
#     train_score += r[0]
# for r in test_losses:
#     #print("Test loss with mse cloning = %f" % r[0])
#     #print("Score with selfrolled mse = %f" % r[1])
#     #print("Performance with selfrolled policy: %d" % r[2])
#     test_score += r[0] #traditional mse
#     p += r[2]
# train_score /= len(train_losses)
# test_score /= len(test_losses)
# p /= len(test_losses)
# lox = results 
# with open("ncp_bc_%db_%de_alone_log.txt"%(job_data['bc_batch_size'], job_data['bc_epochs']), 'a') as log_file:
#     for lo in lox:
#         log_file.write("%f\n"%lo)
#     # log_file.write("Training performance: %d / %d\n"%(p, len(test_losses)))#(score[0][4], job_data['eval_rollouts']))
#     # log_file.write("Average training loss: %f\n"%train_score)
#     # log_file.write("Average eval loss: %f"%test_score)

# r = pd.read_csv("log/default/version_2/metrics.csv")
# grouped_lox = r.groupby('epoch')
# mean_lox = grouped_lox.mean()
# mean_lox = mean_lox.reset_index()
# for i in mean_lox:
#     if i == "train_loss":
#         lox = mean_lox[i].tolist()
# plt.plot(lox)
# plt.title("NCP BC Alone")
# plt.xlabel("Epoch")
# plt.ylabel("Batch Loss")
# plt.savefig("ncp_bc_%db_%de_alone_log.png"%(job_data['bc_batch_size'], job_data['bc_epochs']))
# plt.close()

# if job_data['algorithm'] != 'DAPG':
#     # We throw away the demo data when training from scratch or fine-tuning with RL without explicit augmentation
#     demo_paths = None

#pickle.dump(policy, open("ncp_bc_%db_%de_alone_log.pickle"%(job_data['bc_batch_size'], job_data['bc_epochs']), 'wb'))
# print(train_score)
# print(test_score)

#policy.print_param_values()
#raise Exception
# plt.plot(range(2, 2+tested_layers), avg_train_rewards_n, label="Train Loss")
# plt.plot(range(2, 2+tested_layers), avg_test_rewards_n, label="Test Loss")
# plt.title("RNN BC Depth vs Train/Test Loss")
# plt.xlabel("# of Layers")
# plt.ylabel("Avg. Loss")
# plt.legend()
# plt.savefig("rnn_bc_%d-%dn_alone_graph.png"%(init_layers, init_layers + tested_layers))
# plt.close()



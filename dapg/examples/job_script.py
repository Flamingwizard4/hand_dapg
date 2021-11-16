"""
This is a job script for running NPG/DAPG on hand tasks and other gym envs.
Note that DAPG generalizes PG and BC init + PG finetuning.
With appropriate settings of parameters, we can recover the full family.
"""

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.algos.dapg import DAPG
from mjrl.algos.behavior_cloning import BC
from mjrl.utils.train_agent import train_agent
from mjrl.samplers.core import sample_paths
import os
import json
import mjrl.envs
import mj_envs
import time as timer
import pickle
import argparse
from matplotlib import pyplot as plt

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

# ===============================================================================
# Train Loop
# ===============================================================================

e = GymEnv(job_data['env'])
'''
avg_train_reward_b, avg_test_reward_b = [], []
init_batch = 1
tested_batches = 4
for b in [2**x for x in range(init_batch, init_batch+tested_batches)]:

    policy = MLP(e.spec, hidden_sizes=(64,64), seed=job_data['seed'])
    baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=job_data['vf_batch_size'],
                        epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'])

    # Get demonstration data if necessary and behavior clone
    if job_data['algorithm'] != 'NPG':
        print("========================================")
        print("Collecting expert demonstrations")
        print("========================================")
        demo_paths = pickle.load(open(job_data['demo_file'], 'rb'))

        bc_agent = BC(demo_paths[:-5], policy=policy, env=e, epochs=job_data['bc_epochs'], batch_size=b,
                    lr=job_data['bc_learn_rate'], loss_type='MSE', set_transforms=False)
        in_shift, in_scale, out_shift, out_scale = bc_agent.compute_transformations()
        bc_agent.set_transformations(in_shift, in_scale, out_shift, out_scale)
        bc_agent.set_variance_with_data(out_scale)

        ts = timer.time()
        print("========================================")
        print("Running BC with expert demonstrations")
        print("========================================")
        lv = bc_agent.train()
        print("========================================")
        print("BC training complete !!!")
        print("time taken = %f" % (timer.time() - ts))
        print("========================================")

        # with open("bc_alone_4n_losses.txt", 'a') as out_file:
        #     for l iplease back away and let me gon lv:
        #         out_file.write("%f\n"%l)
        
        train_score, test_score, p = 0, 0, 0
        if job_data['eval_rollouts'] >= 1:
            #score = e.evaluate_policy(policy, num_episodes=job_data['eval_rollouts'], mean_action=True)
            #print(batchrain_losses:
            #print("Training loss with mse cloning = %f" % r[0])
            #print("Score with selfrolled mse = %f" % r[1])
            #print("Performance with selfrolled policy: %d" % r[2])
            train_score += r[0]
        for r in test_losses:
            #print("Test loss with mse cloning = %f" % r[0])
            #print("Score with selfrolled mse = %f" % r[1])
            #print("Performance with selfrolled policy: %d" % r[2])
            test_score += r[0] #traditional mse
            p += r[2]
        train_score /= len(train_losses)
        test_score /= len(test_losses)
        p /= len(test_losses)
    with open("bc_%dn_%db_%de_alone.txt"%(job_data['bc_n_layers'], b, job_data['bc_epochs']), 'a') as log_file:
        for lo in lv:
            log_file.write("%f\n"%lo)
        log_file.write("Test performance: %d / %d\n"%(p, len(test_losses)))#(score[0][4], job_data['eval_rollouts']))
        log_file.write("Average trainig loss: %f\n"%train_score)
        log_file.write("Average eval loss: %f"%test_score)
    plt.plot(lv)
    plt.title("MLP BC Alone w/ Batch Size of %d"%b)
    plt.xlabel("Epoch")
    plt.ylabel("Batch Loss")
    plt.savefig("bc_%dn_%db_%de_alone_log.png"%(job_data['bc_n_layers'], b, job_data['bc_epochs']))
    plt.close()

    if job_data['algorithm'] != 'DAPG':
        # We throw away the demo data when training from scratch or fine-tuning with RL without explicit augmentation
        demo_paths = None

    pickle.dump(policy, open('bc_%dn_%db_%de_alone.pickle'%(job_data['bc_n_layers'], b, job_data['bc_epochs']), 'wb'))
    avg_train_reward.append(train_score)
    avg_test_reward.append(test_score)

plt.plot([2**batch for batch in range(init_batch, init_batch+tested_batches)], avg_train_reward, label="Train Loss")
plt.plot([2**batch for batch in range(init_batch, init_batch+tested_batches)], avg_test_reward, label="Test Loss")
plt.title("MLP BC Batch Size vs Train/Test Loss")
plt.xlabel("Batch Size")
#plt.xscale('linear')
plt.ylabel("Average Loss")
plt.legend()
plt.savefig("mlp_bc_%d-%db_alone_graph.png"%(2, 2**tested_batches))
plt.close()
'''

avg_train_reward_n, avg_test_reward_n = [], []
init_layer = 2
tested_layers = 2
for n in range(init_layer, init_layer+tested_layers): #2+ layers
    hz = job_data['bc_hidden_size']
    hidden_size = [hz, hz]
    for i in range(2, n):
        hidden_size += [hz]
    hidden_size = tuple(hidden_size)
    policy = MLP(e.spec, hidden_sizes=hidden_size, seed=job_data['seed'])
    baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=job_data['vf_batch_size'],
                        epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'])

    # Get demonstration data if necessary and behavior clone
    if job_data['algorithm'] != 'NPG':
        print("========================================")
        print("Collecting expert demonstrations")
        print("========================================")
        demo_paths = pickle.load(open(job_data['demo_file'], 'rb'))
        bc_agent = BC(demo_paths[:-5], policy=policy, env=e, epochs=job_data['bc_epochs'], batch_size=job_data['bc_batch_size'],
                    lr=job_data['bc_learn_rate'], loss_type='MSE', set_transforms=False)
        in_shift, in_scale, out_shift, out_scale = bc_agent.compute_transformations()
        bc_agent.set_transformations(in_shift, in_scale, out_shift, out_scale)
        bc_agent.set_variance_with_data(out_scale)

        ts = timer.time()
        print("========================================")
        print("Running %d-layer BC with expert demonstrations"%n)
        print("========================================")
        lv = bc_agent.train()
        print("========================================")
        print("BC training complete !!!")
        print("time taken = %f" % (timer.time() - ts))
        print("========================================")
        train_score, test_score, p = 0, 0, 0
        train_losses, test_losses = e.evaluate_train_vs_test(bc_agent, demo_paths)
        for r in train_losses:
            #print("Test loss with mse cloning = %f" % r[0])
            #print("Training loss with mse cloning = %f" % r[0])
            #print("Score with selfrolled mse = %f" % r[1])(64,64)
            train_score += r[0]
        for r in test_losses:
            #print("Test loss with mse cloning = %f" % r[0])
            #print("Score with selfrolled mse = %f" % r[1])
            #print("Performance with selfrolled policy: %d" % r[2])
            test_score += r[0] #traditional mse
            p += r[2]
        train_score /= len(train_losses)
        test_score /= len(test_losses)
        p /= len(test_losses)
    with open("bc_%dn_%db_%de_alone.txt"%(n, job_data['bc_batch_size'], job_data['bc_epochs']), 'a') as log_file:
        for lo in lv:
            log_file.write("%f\n"%lo)
        log_file.write("Test performance: %d / %d\n"%(p, len(test_losses)))
        log_file.write("Average training loss: %f\n"%train_score)
        log_file.write("Average eval loss: %f"%test_score)
    plt.plot(lv)
    plt.title("MLP BC Alone w/ %d # of Layers"%n)
    plt.xlabel("Epoch")
    plt.ylabel("Batch Loss")
    plt.savefig("bc_%dn_%db_%de_alone_log.png"%(n, job_data['bc_batch_size'], job_data['bc_epochs']))
    plt.close()

    if job_data['algorithm'] != 'DAPG':
        # We throw away the demo data when training from scratch or fine-tuning with RL without explicit augmentation
        demo_paths = None

    pickle.dump(policy, open('bc_%dn_%db_%de_alone.pickle'%(n, job_data['bc_batch_size'], job_data['bc_epochs']), 'wb'))
    avg_train_reward_n.append(train_score)
    avg_test_reward_n.append(test_score)

plt.plot([l for l in range(init_layer, init_layer+tested_layers)], avg_train_reward_n, label="Train Loss")
plt.plot([l for l in range(init_layer, init_layer+tested_layers)], avg_test_reward_n, label="Test Loss")
plt.title("MLP BC Batch Size vs Train/Test Loss")
plt.xlabel("Batch Size")
#plt.xscale('linear')
plt.ylabel("Average Loss")
plt.legend()
plt.savefig("mlp_bc_%d-%db_alone_graph.png"%(init_layer, init_layer + tested_layers - 1))
plt.close()
'''
# ===============================================================================
# RL Loop
# ===============================================================================

rl_agent = DAPG(e, policy, baseline, demo_paths,
                normalized_step_size=job_data['rl_step_size'],
                lam_0=job_data['lam_0'], lam_1=job_data['lam_1'],
                seed=job_data['seed'], save_logs=True
                )

print("========================================")
print("Starting reinforcement learning phase")
print("========================================")

ts = timer.time()
train_agent(job_name=JOB_DIR,
            agent=rl_agent,
            seed=job_data['seed'],
            niter=job_data['rl_num_iter'],
            gamma=job_data['rl_gamma'],
            gae_lambda=job_data['rl_gae'],
            num_cpu=job_data['num_cpu'],
            sample_mode='trajectories',
            num_traj=job_data['rl_num_traj'],
            save_freq=job_data['save_freq'],
            evaluation_rollouts=job_data['eval_rollouts'])
print("time taken = %f" % (timer.time()-ts))
'''
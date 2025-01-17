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
baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=job_data['vf_batch_size'],
                       epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'])
scores, avg_r = [], []
max_layers = 5
for n in range(1, max_layers+1): #this loop can be for hidden size, n_layers, batch size, and epochs
    hs = int(job_data["hidden_size"])
    hidden_size = [hs]
    if n > 1:
        for n_i in range(1, n):
            hidden_size.append(hs)
    hidden_size = tuple(hidden_size)
    policy = MLP(e.spec, hidden_sizes=hidden_size, seed=job_data['seed']) #
    # Get demonstration data if necessary and behavior clone
    if job_data['algorithm'] != 'NPG':
        print("========================================")
        print("Collecting expert demonstrations")
        print("========================================")
        demo_paths = pickle.load(open(job_data['demo_file'], 'rb'))

        bc_agent = BC(demo_paths, policy=policy, epochs=job_data['bc_epochs'], batch_size=job_data['bc_batch_size'],
                    lr=job_data['bc_learn_rate'], loss_type='MSE', set_transforms=False)
        in_shift, in_scale, out_shift, out_scale = bc_agent.compute_transformations()
        bc_agent.set_transformations(in_shift, in_scale, out_shift, out_scale)
        bc_agent.set_variance_with_data(out_scale)

        ts = timer.time()
        print("========================================")
        print("Running %d-layer MLP BC with expert demonstrations"%n)
        print("========================================")
        bc_agent.train()
        lox = bc_agent.losses
        print("========================================")
        print("BC training complete !!!")
        print("time taken = %f" % (timer.time() - ts))
        print("========================================")

        if job_data['eval_rollouts'] >= 1:
            score = e.evaluate_policy(policy, num_episodes=job_data['eval_rollouts'], mean_action=True)
            print("Score with behavior cloning = %f" % score[0][0])
            print("Performance with BC: %d / %d"%(score[0][4], job_data['eval_rollouts']))
            print("Average reward: %f"%score[0][5])
            scores.append(score)
    with open("mlp_bc_%dn_alone_log.txt"%n, 'a') as log_file:
        for lo in lox:
            log_file.write("%f\n"%lo)
        log_file.write("Total performance: %d / %d"%(score[0][4], job_data['eval_rollouts']))
        log_file.write("Average reward: %f"%score[0][5])
    plt.plot(lox)
    plt.title("MLP BC Alone w/ %d Layers"%n)
    plt.xlabel("Epoch")
    plt.ylabel("Batch Loss")
    plt.savefig("mlp_bc_%dn_alone_log.png"%n)
    plt.close()
    if job_data['algorithm'] != 'DAPG':
        # We throw away the demo data when training from scratch or fine-tuning with RL without explicit augmentation
        demo_paths = None

    pickle.dump(policy, open('mlp_bc_%dn_alone.pickle'%n, 'wb'))
    avg_r.append(score[0][5])
plt.plot(range(1, max_layers+1),avg_r)
plt.title("MLP BC Depth vs Reward")
plt.xlabel("# of Layers")
plt.ylabel("Average Reward")
plt.savefig("mlp_bc_%d-%dn_alone_graph.png"%(n, max_layers))
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
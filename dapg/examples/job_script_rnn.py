"""
This is a job script for running NPG/DAPG on hand tasks and other gym envs.
Note that DAPG generalizes PG and BC init + PG finetuning.
With appropriate settings of parameters, we can recover the full family.
"""

from mjrl.utils.gym_env_rnn import GymEnv
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
'''max_layers = 5
avg_rewards_n = []
for n in range(1, max_layers+1): #n_layers
    policy = RNN(e.spec, n_layers=n, seed=job_data['seed']) #job_data['policy_size']
    #policy = MLP(e.spec, hidden_sizes=job_data['policy_size'], seed=job_data['seed'])
    #baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=job_data['vf_batch_size'],
                        #epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'])

    # Get demonstration data if necessary and behavior clone
    if job_data['algorithm'] != 'NPG':
        print("========================================")
        print("Collecting expert demonstrations")
        print("========================================")
        demo_paths = pickle.load(open(job_data['demo_file'], 'rb'))
        bc_agent = rnn_BC(demo_paths, policy=policy, epochs=job_data['bc_epochs'], seed=job_data['seed'], batch_size=job_data["bc_batch_size"],
                    lr=job_data['bc_learn_rate'], loss_type='MSE', set_transforms=False)
        in_shift, in_scale, out_shift, out_scale = bc_agent.compute_transformations()
        bc_agent.set_transformations(in_shift, in_scale, out_shift, out_scale)
        bc_agent.set_variance_with_data(out_scale)

        ts = timer.time()
        print("========================================")
        print("Running %d-layer RNN BC with expert demonstrations"%n)
        print("========================================")
        lox = bc_agent.train()
        print("========================================")
        print("BC training complete !!!")
        print("time taken = %f" % (timer.time() - ts))
        print("========================================")

        if job_data['eval_rollouts'] >= 1:
            score = e.evaluate_policy(policy, num_episodes=job_data['eval_rollouts'], mean_action=True)
            print("Score with behavior cloning = %f" % score[0][0])
            print("Performance with BC: %d / %d"%(score[0][4], job_data['eval_rollouts']))
    with open("rnn_bc_%dn_alone_log.txt"%n, 'a') as log_file:
        for lo in lox:
            log_file.write("%f\n"%lo)
        log_file.write("Total performance: %d / %d"%(score[0][4], job_data['eval_rollouts']))
        log_file.write("Average reward: %f"%score[0][0])
    plt.plot(lox)
    plt.title("RNN BC Alone w/ %d Layers"%n)
    plt.xlabel("Epoch")
    plt.ylabel("Batch Loss")
    plt.savefig("rnn_bc_%dn_alone_log.png"%n)
    plt.close()
    if job_data['algorithm'] != 'DAPG':
        # We throw away the demo data when training from scratch or fine-tuning with RL without explicit augmentation
        demo_paths = None

    pickle.dump(policy, open('rnn_bc_%dn_alone.pickle'%n, 'wb'))
    avg_rewards_n.append(score[0][0])

    #policy.print_param_values()
    #raise Exception
plt.plot(range(1, max_layers+1),avg_rewards_n)
plt.title("RNN BC Depth vs Reward")
plt.xlabel("# of Layers")
plt.ylabel("Average Reward")
plt.savefig("rnn_bc_%d-%dn_alone_graph.png"%(1, max_layers))
plt.close()
'''
max_batch_size = 40
avg_rewards_b = []
### Varying Batch Size
#for b in range(8, max_batch_size+1, 8):
for b in [x**2 for x in range(2, 7)]:
    policy = RNN(e.spec, n_layers=2, seed=job_data['seed']) #job_data['policy_size']
    #policy = MLP(e.spec, hidden_sizes=job_data['policy_size'], seed=job_data['seed'])
    #baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=job_data['vf_batch_size'],
                        #epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'])

    # Get demonstration data if necessary and behavior clone
    if job_data['algorithm'] != 'NPG':
        print("========================================")
        print("Collecting expert demonstrations")
        print("========================================")
        demo_paths = pickle.load(open(job_data['demo_file'], 'rb'))
        bc_agent = rnn_BC(demo_paths, policy=policy, epochs=job_data['bc_epochs'], seed=job_data['seed'], batch_size=b,
                    lr=job_data['bc_learn_rate'], loss_type='MSE', set_transforms=False)
        in_shift, in_scale, out_shift, out_scale = bc_agent.compute_transformations()
        bc_agent.set_transformations(in_shift, in_scale, out_shift, out_scale)
        bc_agent.set_variance_with_data(out_scale)

        ts = timer.time()
        print("========================================")
        print("Running %d-batch size RNN BC with expert demonstrations"%b)
        print("========================================")
        lox = bc_agent.train()
        print("========================================")
        print("BC training complete !!!")
        print("time taken = %f" % (timer.time() - ts))
        print("========================================")

        if job_data['eval_rollouts'] >= 1:
            score = e.evaluate_policy(policy, num_episodes=job_data['eval_rollouts'], mean_action=True, seed=321)
            print("Score with behavior cloning = %f" % score[0][0])
            print("Performance with BC: %d / %d"%(score[0][4], job_data['eval_rollouts']))
    with open("rnn_bc_%db_alone_log.txt"%b, 'a') as log_file:
        for lo in lox:
            log_file.write("%f\n"%lo)
        log_file.write("Total performance: %d / %d"%(score[0][4], job_data['eval_rollouts']))
        log_file.write("Average reward: %f"%score[0][0])
    plt.plot(lox)
    plt.title("RNN BC Alone w/ Batch Size of %d"%b)
    plt.xlabel("Epoch")
    plt.ylabel("Batch Loss")
    plt.savefig("rnn_bc_%db_alone_log.png"%b)
    plt.close()
    if job_data['algorithm'] != 'DAPG':
        # We throw away the demo data when training from scratch or fine-tuning with RL without explicit augmentation
        demo_paths = None

    pickle.dump(policy, open('rnn_bc_%db_alone.pickle'%b, 'wb'))
    avg_rewards_b.append(score[0][0])

#plt.plot([batch for batch in range(8, max_batch_size+1, 8)], avg_rewards_b)
plt.plot([batch**2 for batch in range(2, 7)], avg_rewards_b)
plt.title("RNN BC Batch Size vs Reward")
plt.xlabel("Batch Size")
#plt.xscale('linear')
plt.ylabel("Average Reward")
#plt.savefig("rnn_bc_%d-%db_alone_graph.png"%(8, max_batch_size))
plt.savefig("rnn_bc_%d-%db_alone_graph.png"%(4, 64))
plt.close()
'''
tested_epochs = 3
avg_rewards_t = []
### Varying Training Epochs
for t in [500*(2**x) for x in range(tested_epochs)]:
    policy = RNN(e.spec, n_layers=2, seed=job_data['seed']) #job_data['policy_size']
    #policy = MLP(e.spec, hidden_sizes=job_data['policy_size'], seed=job_data['seed'])
    #baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=job_data['vf_batch_size'],
                        #epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'])

    # Get demonstration data if necessary and behavior clone
    if job_data['algorithm'] != 'NPG':
        print("========================================")
        print("Collecting expert demonstrations")
        print("========================================")
        demo_paths = pickle.load(open(job_data['demo_file'], 'rb'))
        bc_agent = rnn_BC(demo_paths, policy=policy, epochs=t, seed=job_data['seed'], batch_size=job_data['bc_batch_size'],
                    lr=job_data['bc_learn_rate'], loss_type='MSE', set_transforms=False)
        in_shift, in_scale, out_shift, out_scale = bc_agent.compute_transformations()
        bc_agent.set_transformations(in_shift, in_scale, out_shift, out_scale)
        bc_agent.set_variance_with_data(out_scale)

        ts = timer.time()
        print("========================================")
        print("Running RNN BC with expert demonstrations for %d epochs"%t)
        print("========================================")
        lox = bc_agent.train()
        print("========================================")
        print("BC training complete !!!")
        print("time taken = %f" % (timer.time() - ts))
        print("========================================")

        if job_data['eval_rollouts'] >= 1:
            score = e.evaluate_policy(policy, num_episodes=job_data['eval_rollouts'], mean_action=True)
            print("Score with behavior cloning = %f" % score[0][0])
            print("Performance with BC: %d / %d"%(score[0][4], job_data['eval_rollouts']))
    with open("rnn_bc_%dt_alone_log.txt"%t, 'a') as log_file:
        for lo in lox:
            log_file.write("%f\n"%lo)
        log_file.write("Total performance: %d / %d"%(score[0][4], job_data['eval_rollouts']))
        log_file.write("Average reward: %f"%score[0][0])
    plt.plot(lox)
    plt.title("RNN BC Alone w/ %d Epochs"%t)
    plt.xlabel("Epoch")
    plt.xscale('linear')
    plt.ylabel("Batch Loss")
    plt.savefig("rnn_bc_%dt_alone_log.png"%t)
    plt.close()
    if job_data['algorithm'] != 'DAPG':
        # We throw away the demo data when training from scratch or fine-tuning with RL without explicit augmentation
        demo_paths = None

    pickle.dump(policy, open('rnn_bc_%dt_alone.pickle'%t, 'wb'))
    avg_rewards_t.append(score[0][0])

plt.plot([80*(2**x) for x in range(tested_epochs)], avg_rewards_t)
plt.title("RNN BC Training Duration vs Reward")
plt.xlabel("Training Epochs")
plt.xscale('log')
plt.ylabel("Average Reward")
plt.savefig("rnn_bc_%d-%dt_alone_graph.png"%(80, 80*2**(tested_epochs-1)))
plt.close()
'''
'''
tested_rates = 5
avg_rewards_l = []
### Varying Learning Rates
for l in [(0.0001*(10**(x/2)) if x % 2 == 0 else 0.0005*(10**(x//2))) for x in range(tested_rates)]:
    policy = RNN(e.spec, n_layers=2, seed=job_data['seed']) #job_data['policy_size']
    #policy = MLP(e.spec, hidden_sizes=job_data['policy_size'], seed=job_data['seed'])
    #baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=job_data['vf_batch_size'],
                        #epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'])

    # Get demonstration data if necessary and behavior clone
    if job_data['algorithm'] != 'NPG':
        print("========================================")
        print("Collecting expert demonstrations")
        print("========================================")
        demo_paths = pickle.load(open(job_data['demo_file'], 'rb'))
        bc_agent = rnn_BC(demo_paths, policy=policy, epochs=job_data['bc_epochs'], seed=job_data['seed'], batch_size=job_data['bc_batch_size'],
                    lr=l, loss_type='MSE', set_transforms=False)
        in_shift, in_scale, out_shift, out_scale = bc_agent.compute_transformations()
        bc_agent.set_transformations(in_shift, in_scale, out_shift, out_scale)
        bc_agent.set_variance_with_data(out_scale)

        ts = timer.time()
        print("========================================")
        print("Running %f-lr RNN BC with expert demonstrations"%l)
        print("========================================")
        lox = bc_agent.train()
        print("========================================")
        print("BC training complete !!!")
        print("time taken = %f" % (timer.time() - ts))
        print("========================================")

        if job_data['eval_rollouts'] >= 1:
            score = e.evaluate_policy(policy, num_episodes=job_data['eval_rollouts'], mean_action=True)
            print("Score with behavior cloning = %f" % score[0][0])
            print("Performance with BC: %d / %d"%(score[0][4], job_data['eval_rollouts']))
    with open("rnn_bc_%flr_alone_log.txt"%l, 'a') as log_file:
        for lo in lox:
            log_file.write("%f\n"%lo)
        log_file.write("Total performance: %d / %d"%(score[0][4], job_data['eval_rollouts']))
        log_file.write("Average reward: %f"%score[0][0])
    plt.plot(lox)
    plt.title("RNN BC Alone w/ %f Learning Rate"%l)
    plt.xlabel("Epoch")
    plt.xscale('linear')
    plt.ylabel("Batch Loss")
    plt.savefig("rnn_bc_%flr_alone_log.png"%l)
    plt.close()
    if job_data['algorithm'] != 'DAPG':
        # We throw away the demo data when training from scratch or fine-tuning with RL without explicit augmentation
        demo_paths = None

    pickle.dump(policy, open('rnn_bc_%flr_alone.pickle'%l, 'wb'))
    avg_rewards_l.append(score[0][0])

plt.plot([(0.0001*(10**(x/2)) if x % 2 == 0 else 0.0005*(10**(x//2))) for x in range(tested_rates)], avg_rewards_l)
plt.title("RNN BC LR vs Reward")
plt.xlabel("Learning Rate")
plt.xscale('log')
plt.ylabel("Average Reward")
plt.savefig("rnn_bc_%f-%flr_alone_graph.png"%(0.0001, 0.0001*10**(tested_rates-1)))
plt.close()

'''
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
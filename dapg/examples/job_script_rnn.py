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
###Varying Number of Layers
init_layers = 2
tested_layers = 4
avg_train_rewards_n, avg_test_rewards_n = [], []
for n in range(init_layers, init_layers + tested_layers): #n_layers
    policy = RNN(e.spec, hidden_size=job_data['bc_hidden_size'], n_layers=n, seed=job_data['seed']) #job_data['policy_size']
    #policy = MLP(e.spec, hidden_sizes=job_data['policy_size'], seed=job_data['seed'])
    #baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=job_data['vf_batch_size'],
                        #epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'])

    # Get demonstration data if necessary and behavior clone
    if job_data['algorithm'] != 'NPG':
        print("========================================")
        print("Collecting expert demonstrations")
        print("========================================")
        demo_paths = pickle.load(open(job_data['demo_file'], 'rb'))
        bc_agent = rnn_BC(demo_paths, policy=policy, env=e, epochs=job_data['bc_epochs'], seed=job_data['seed'], batch_size=job_data["bc_batch_size"],
                    lr=job_data['bc_learn_rate'], loss_type='MSE', set_transforms=False)
        in_shift, in_scale, out_shift, out_scale = bc_agent.compute_transformations()
        bc_agent.set_transformations(in_shift, in_scale, out_shift, out_scale)
        bc_agent.set_variance_with_data(out_scale)

        ts = timer.time()
        print("========================================")
        print("Running %d-layer, %d-batch size RNN BC with expert demonstrations for %d epochs"%(n, job_data["bc_batch_size"], job_data['bc_epochs']))
        print("========================================")
        lox = bc_agent.train()
        print("========================================")
        print("BC training complete !!!")
        print("time taken = %f" % (timer.time() - ts))
        print("========================================")

        # if job_data['eval_rollouts'] >= 1:
        #     score = e.evaluate_policy(policy, num_episodes=job_data['eval_rollouts'], mean_action=True, seed=321)
        #     print("Score with behavior cloning = %f" % score[0][0])
        #     print("Performance with BC: %d / %d"%(score[0][4], job_data['eval_rollouts']))
        train_losses, test_losses = e.evaluate_train_vs_test(bc_agent, demo_paths)
        train_score, test_score, p = 0, 0, 0
        for r in train_losses:
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
    with open("rnn_bc_%dn_%db_%de_alone_log.txt"%(n, job_data['bc_batch_size'], job_data['bc_epochs']), 'a') as log_file:
        for lo in lox:
            log_file.write("%f\n"%lo)
        log_file.write("Training performance: %d / %d\n"%(p, len(test_losses)))#(score[0][4], job_data['eval_rollouts']))
        log_file.write("Average training loss: %f\n"%train_score)
        log_file.write("Average eval loss: %f"%test_score)
    
    plt.plot(lox)
    plt.title("RNN BC Alone w/ %d Layers"%n)
    plt.xlabel("Epoch")
    plt.ylabel("Batch Loss")
    plt.savefig("rnn_bc_%dn_%db_%de_alone_log.png"%(n, job_data['bc_batch_size'], job_data['bc_epochs']))
    plt.close()
    if job_data['algorithm'] != 'DAPG':
        # We throw away the demo data when training from scratch or fine-tuning with RL without explicit augmentation
        demo_paths = None

    pickle.dump(policy, open('rnn_bc_%dn_%db_%de_alone.pickle'%(n, job_data['bc_batch_size'], job_data['bc_epochs']), 'wb'))
    avg_train_rewards_n.append(train_score)
    avg_test_rewards_n.append(test_score)

    #policy.print_param_values()
    #raise Exception
plt.plot(range(2, 2+tested_layers), avg_train_rewards_n, label="Train Loss")
plt.plot(range(2, 2+tested_layers), avg_test_rewards_n, label="Test Loss")
plt.title("RNN BC Depth vs Train/Test Loss")
plt.xlabel("# of Layers")
plt.ylabel("Avg. Loss")
plt.savefig("rnn_bc_%d-%dn_alone_graph.png"%(init_layers, init_layers + tested_layers))
plt.close()
'''
### Varying Batch Size
init_batch = 1
avg_train_rewards_b, avg_test_rewards_b = [], []
tested_batches = 4
#for b in range(8, max_batch_size+1, 8):
for b in [2**x for x in range(init_batch, init_batch+tested_batches)]:
    policy = RNN(e.spec, hidden_size=job_data['bc_hidden_size'], n_layers=job_data['bc_n_layers'], seed=job_data['seed']) #job_data['hidden_size']
    #policy = MLP(e.spec, hidden_sizes=job_data['policy_size'], seed=job_data['seed'])
    #baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=job_data['vf_batch_size'],
                        #epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'])

    # Get demonstration data if necessary and behavior clone
    if job_data['algorithm'] != 'NPG':
        print("========================================")
        print("Collecting expert demonstrations")
        print("========================================")
        demo_paths = pickle.load(open(job_data['demo_file'], 'rb'))
        bc_agent = rnn_BC(demo_paths[:-5], policy=policy, env=e, epochs=job_data['bc_epochs'], seed=job_data['seed'], batch_size=b,
                    lr=job_data['bc_learn_rate'], loss_type='MSE', set_transforms=False)
        in_shift, in_scale, out_shift, out_scale = bc_agent.compute_transformations()
        bc_agent.set_transformations(in_shift, in_scale, out_shift, out_scale)
        bc_agent.set_variance_with_data(out_scale)

        ts = timer.time()
        print("========================================")
        print("Running %d-layer, %d-batch size RNN BC with expert demonstrations for %d epochs"%(job_data['bc_n_layers'], b, job_data['bc_epochs']))
        print("========================================")
        lox = bc_agent.train()
        print("========================================")
        print("BC training complete !!!")
        print("time taken = %f" % (timer.time() - ts))
        print("========================================")

        train_losses, test_losses = e.evaluate_train_vs_test(bc_agent, demo_paths)
        train_score, test_score, p = 0, 0, 0
        for r in train_losses:
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
    with open("rnn_bc_%dn_%db_%de_alone_log.txt"%(job_data['bc_n_layers'], b, job_data['bc_epochs']), 'a') as log_file:
        for lo in lox:
            log_file.write("%f\n"%lo)
        log_file.write("Training performance: %d / %d\n"%(p, len(test_losses)))#(score[0][4], job_data['eval_rollouts']))
        log_file.write("Average training loss: %f\n"%train_score)
        log_file.write("Average eval loss: %f"%test_score)
    plt.plot(lox)
    plt.title("RNN BC Alone w/ Batch Size of %d"%b)
    plt.xlabel("Epoch")
    plt.ylabel("Batch Loss")
    plt.savefig("rnn_bc_%dn_%db_%de_alone_log.png"%(job_data['bc_n_layers'], b, job_data['bc_epochs']))
    plt.close()

    if job_data['algorithm'] != 'DAPG':
        # We throw away the demo data when training from scratch or fine-tuning with RL without explicit augmentation
        demo_paths = None

    pickle.dump(policy, open('rnn_bc_%dn_%db_%de_alone.pickle'%(job_data['bc_n_layers'], b, job_data['bc_epochs']), 'wb'))
    avg_train_rewards_b.append(train_score)
    avg_test_rewards_b.append(test_score)
    
#plt.plot([batch for batch in range(8, max_batch_size+1, 8)], avg_rewards_b)
plt.plot([2**batch for batch in range(init_batch, init_batch+tested_batches)], avg_train_rewards_b, label="Train Loss")
plt.plot([2**batch for batch in range(init_batch, init_batch+tested_batches)], avg_test_rewards_b, label="Test Loss")
plt.title("RNN BC Batch Size vs Train/Test Loss")
plt.xlabel("Batch Size")
#plt.xscale('linear')
plt.ylabel("Average Loss")
plt.legend()
#plt.savefig("rnn_bc_%d-%db_alone_graph.png"%(8, max_batch_size))
plt.savefig("rnn_bc_%d-%db_alone_graph.png"%(2, 2**tested_batches))
plt.close()
'''
'''
### Varying Training Epochs
init_epochs = 20
tested_epochs = 3
avg_train_rewards_t, avg_test_rewards_t = [], []
for t in [init_epochs*(2**x) for x in range(tested_epochs)]:
    policy = RNN(e.spec, hidden_size=job_data['bc_hidden_size'], n_layers=job_data['bc_n_layers'], seed=job_data['seed'])
    #policy = MLP(e.spec, hidden_sizes=job_data['policy_size'], seed=job_data['seed'])
    #baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=job_data['vf_batch_size'],
                        #epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'])

    # Get demonstration data if necessary and behavior clone
    if job_data['algorithm'] != 'NPG':
        print("========================================")
        print("Collecting expert demonstrations")
        print("========================================")
        demo_paths = pickle.load(open(job_data['demo_file'], 'rb'))
        bc_agent = rnn_BC(demo_paths[:-5], policy, e, epochs=t, seed=job_data['seed'], batch_size=job_data['bc_batch_size'],
                    lr=job_data['bc_learn_rate'], loss_type='MSE', set_transforms=False)
        in_shift, in_scale, out_shift, out_scale = bc_agent.compute_transformations()
        bc_agent.set_transformations(in_shift, in_scale, out_shift, out_scale)
        bc_agent.set_variance_with_data(out_scale)

        ts = timer.time()
        print("========================================")
        print("Running %d-layer, %d-batch size RNN BC with expert demonstrations for %d epochs"%(job_data['bc_n_layers'], job_data['bc_batch_size'], t))
        print("========================================")
        lox = bc_agent.train()
        print("========================================")
        print("BC training complete !!!")
        print("time taken = %f" % (timer.time() - ts))
        print("========================================")

        # if job_data['eval_rollouts'] >= 1:
        #     score = e.evaluate_train_vs_test(policy, demo_paths, num_episodes=job_data['eval_rollouts'], mean_action=True, seed=321)
        #     print("Score with behavior cloning = %f" % score[0][0])
        #     print("Performance with BC: %d / %d"%(score[0][4], job_data['eval_rollouts']))
        train_losses, test_losses = e.evaluate_train_vs_test(bc_agent, demo_paths)
        train_score, test_score, p = 0, 0, 0
        for r in train_losses:
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
    with open("rnn_bc_%dn_%db_%de_alone_log.txt"%(job_data['bc_n_layers'], job_data["bc_batch_size"], t), 'a') as log_file:
        for lo in lox:
            log_file.write("%f\n"%lo)
        log_file.write("Training performance: %d / %d\n"%(p, len(test_losses)))#(score[0][4], job_data['eval_rollouts']))
        log_file.write("Average training loss: %f\n"%train_score)
        log_file.write("Average eval loss: %f"%test_score)
    
    plt.plot(lox)
    plt.title("RNN BC Alone w/ %d Epochs"%t)
    plt.xlabel("Epoch")
    plt.xscale('linear')
    plt.ylabel("Batch Loss")
    plt.savefig("rnn_bc_%dn_%db_%de_alone_log.png"%(job_data['bc_n_layers'], job_data["bc_batch_size"], t))
    plt.close()
    if job_data['algorithm'] != 'DAPG':
        # We throw away the demo data when training from scratch or fine-tuning with RL without explicit augmentation
        demo_paths = None

    pickle.dump(policy, open('rnn_bc_%dn_%db_%de_alone.pickle'%(job_data['bc_n_layers'], job_data["bc_batch_size"], t), 'wb'))
    avg_train_rewards_t.append(train_score)
    avg_test_rewards_t.append(test_score)
    

plt.plot([init_epochs*(2**x) for x in range(tested_epochs)], avg_train_rewards_t, label="Train Loss")
plt.plot([init_epochs*(2**x) for x in range(tested_epochs)], avg_test_rewards_t, label="Test Loss")
plt.title("RNN BC Training Epochs vs Test/Train Loss")
plt.xlabel("Training Epochs")
plt.xscale('log')
plt.ylabel("Avg. Loss")
plt.savefig("rnn_bc_%d-%dt_alone_graph.png"%(init_epochs, init_epochs*2**(tested_epochs-1)))
plt.close()
'''
'''
### Varying Learning Rates
tested_rates = 5
avg_train_rewards_l, avg_test_rewards_l = []
for l in [(0.0001*(10**(x/2)) if x % 2 == 0 else 0.0005*(10**(x//2))) for x in range(tested_rates)]:
    policy = RNN(e.spec, hidden_size=job_data['bc_hidden_size'], n_layers=job_data['bc_n_layers'], seed=job_data['seed'])
    #policy = MLP(e.spec, hidden_sizes=job_data['policy_size'], seed=job_data['seed'])
    #baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=job_data['vf_batch_size'],
                        #epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'])

    # Get demonstration data if necessary and behavior clone
    if job_data['algorithm'] != 'NPG':
        print("========================================")
        print("Collecting expert demonstrations")
        print("========================================")
        demo_paths = pickle.load(open(job_data['demo_file'], 'rb'))
        bc_agent = rnn_BC(demo_paths, policy=policy, env=e, epochs=job_data['bc_epochs'], seed=job_data['seed'], batch_size=job_data['bc_batch_size'],
                    lr=l, loss_type='MSE', set_transforms=False)
        in_shift, in_scale, out_shift, out_scale = bc_agent.compute_transformations()
        bc_agent.set_transformations(in_shift, in_scale, out_shift, out_scale)
        bc_agent.set_variance_with_data(out_scale)

        ts = timer.time()
        print("========================================")
        print("Running %f-lr, %d-layer, %d-batch size RNN BC with expert demonstrations for %d epochs"%(l, job_data['bc_n_layers'], job_data['bc_batch_size'], job_data['bc_epochs']))
        print("========================================")
        lox = bc_agent.train()
        print("========================================")
        print("BC training complete !!!")
        print("time taken = %f" % (timer.time() - ts))
        print("========================================")
        # if job_data['eval_rollouts'] >= 1:
        #     score = e.evaluate_policy(policy, num_episodes=job_data['eval_rollouts'], mean_action=True, seed=321)
        #     print("Score with behavior cloning = %f" % score[0][0])
        #     print("Performance with BC: %d / %d"%(score[0][4], job_data['eval_rollouts']))
        train_losses, test_losses = e.evaluate_train_vs_test(bc_agent, demo_paths)
        train_score, test_score, p = 0, 0, 0
        for r in train_losses:
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
    with open("rnn_bc_%flr_%dn_%db_%de_alone_log.txt"%(l, job_data['bc_n_layers'], job_data["bc_batch_size"], job_data['bc_epochs']), 'a') as log_file:
        for lo in lox:
            log_file.write("%f\n"%lo)
        log_file.write("Training performance: %d / %d\n"%(p, len(test_losses)))
        log_file.write("Average training loss: %f\n"%train_score)
        log_file.write("Average eval loss: %f"%test_score)
    
    plt.plot(lox)
    plt.title("RNN BC Alone w/ %f Learning Rate"%l)
    plt.xlabel("Epoch")
    plt.xscale('linear')
    plt.ylabel("Batch Loss")
    plt.savefig("rnn_bc_%flr_%dn_%db_%de_alone.png"%(l, job_data['bc_n_layers'], job_data["bc_batch_size"], job_data['bc_epochs']))
    plt.close()
    if job_data['algorithm'] != 'DAPG':
        # We throw away the demo data when training from scratch or fine-tuning with RL without explicit augmentation
        demo_paths = None

    pickle.dump(policy, open('rnn_bc_%flr_%dn_%db_%de_alone.pickle'%(l, job_data['bc_n_layers'], job_data["bc_batch_size"], job_data['bc_epochs']), 'wb'))
    avg_train_rewards_t.append(train_score)
    avg_test_rewards_t.append(test_score)

plt.plot([(0.0001*(10**(x/2)) if x % 2 == 0 else 0.0005*(10**(x//2))) for x in range(tested_rates)], avg_train_rewards_l)
plt.plot([(0.0001*(10**(x/2)) if x % 2 == 0 else 0.0005*(10**(x//2))) for x in range(tested_rates)], avg_test_rewards_l)
plt.title("RNN BC LR vs Train/Test Loss")
plt.xlabel("Learning Rate")
plt.xscale('log')
plt.ylabel("Avg. Loss")
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
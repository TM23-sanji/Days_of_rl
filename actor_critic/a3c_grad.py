import os, argparse
import gymnasium as gym
import ptan
from ptan.experience import VectorExperienceSourceFirstLast
from ptan.common.utils import TBMeanTracker
from torch.utils.tensorboard.writer import SummaryWriter

import torch
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from lib import common

"""
If GRAD_BATCH = 64 and TRAIN_BATCH = 4:
Worker A finds 64 samples -> calculates delta_A.
Worker B finds 64 samples -> calculates delta_B.
Worker C finds 64 samples -> calculates delta_C.
Worker D finds 64 samples -> calculates delta_D.
Main Process sums delta_A + delta_B + delta_C + delta_D 
and updates the weights using all 256 samples at once.


[ MAIN PROCESS ] 
  |-- 1. net.share_memory() (Shared weights θ)
  |-- 2. Starts 4 Workers
  |
  |  [ Worker 0 ] -> (net) -> [8 Envs] -> Calc Grads_0 -> Put in Queue
  |  [ Worker 1 ] -> (net) -> [8 Envs] -> Calc Grads_1 -> Put in Queue
  |  [ Worker 2 ] -> (net) -> [8 Envs] -> Calc Grads_2 -> Put in Queue
  |  [ Worker 3 ] -> (net) -> [8 Envs] -> Calc Grads_3 -> Put in Queue
  |
  |-- 3. train_queue.get() -> Grads_0
  |-- 4. train_queue.get() -> Grads_1
  |-- 5. SUM (Grads_0 + Grads_1)  <-- Because TRAIN_BATCH = 2
  |-- 6. optimizer.step()        <-- Updates Shared Memory (θ)
  |-- 7. Workers now see new θ automatically because of share_memory()!


  Now talking a bit abt multiprocessing:
  there is a main process and when we do 
  data_proc = mp.Process(target=grads_func, args=p_args)
  data_proc.start()
  
  we create a new process and it has its own memory space
  so we need to share the memory between processes
  
  net.share_memory()    

  now we have shared memory between processes
  so when we update the weights in one process
  the other processes see the updated weights

  also these main process & all the worker processes
  are running in parallel

  so we can have multiple workers
  each with multiple environments

  if we want to pause the main process until say process1 finishes
  we can do process1.join() but rest parallel processes will keep running
"""

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01

REWARD_STEPS = 4
CLIP_GRAD = 0.1

PROCESSES_COUNT = 4
NUM_ENVS = 8

GRAD_BATCH = 64
TRAIN_BATCH = 2


ENV_NAME = "PongNoFrameskip-v4"
NAME = 'pong'
REWARD_BOUND = 18

def make_env() -> gym.Env:
    return ptan.common.wrappers.wrap_dqn(gym.make(ENV_NAME))
    # this wrapper is for basic Atari preprocessing like NoOP reset, frame skipping 

def grads_func(proc_name: str, net: common.AtariA2C, device: torch.device, train_queue: mp.Queue):
    env_factories = [make_env for _ in range(NUM_ENVS)]
    env = gym.vector.SyncVectorEnv(env_factories)
    # see the Sync Keeps the 8 environments inside the same child process. 
    # It runs them in a loop: Step Env 1, then 2, then 3.
    # we want 

    agent = ptan.agent.PolicyAgent(
        lambda x: net(x)[0],
        apply_softmax=True,
        device=device
    )
    exp_source = VectorExperienceSourceFirstLast(
        agent, env, gamma=GAMMA, steps_count=REWARD_STEPS
    )

    batch = []
    frame_idx = 0
    writer = SummaryWriter(comment=f"-{proc_name}")

    with common.RewardTracker(writer, REWARD_BOUND) as tracker:
        with TBMeanTracker(writer, 100) as tb_tracker:
            for exp in exp_source:
                frame_idx += 1
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards and tracker.reward(new_rewards[0], frame_idx):
                    break
                batch.append(exp)
                if len(batch) < GRAD_BATCH:
                    continue

                data = common.unpack_batch(batch, net, device=device, gamma=GAMMA, reward_steps=REWARD_STEPS)
                states_v, actions_t, vals_ref_v = data

                batch.clear()

                net.zero_grad()
                logits_v, value_v = net(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - value_v.detach()
                log_p_a = log_prob_v[range(GRAD_BATCH), actions_t]
                log_prob_actions_v = adv_v * log_p_a
                loss_policy_v = -log_prob_actions_v.mean()
                
                prob_v = F.softmax(logits_v, dim=1)
                ent = (prob_v * log_prob_v).sum(dim=1).mean()
                entropy_loss_v = ENTROPY_BETA * ent
                
                loss_v = entropy_loss_v + loss_policy_v + loss_value_v
                loss_v.backward()
                
                tb_tracker.track("advantage", adv_v, frame_idx)
                tb_tracker.track("values", value_v, frame_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, frame_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, frame_idx)
                tb_tracker.track("loss_policy", loss_policy_v, frame_idx)
                tb_tracker.track("loss_value", loss_value_v, frame_idx)
                tb_tracker.track("loss_total", loss_v, frame_idx)

                # gather gradients
                nn.utils.clip_grad_norm_(
                    net.parameters(), CLIP_GRAD
                )
                grads = [
                    param.grad.data.cpu().numpy() if param.grad is not None else None
                    for param in net.parameters()
                ]
                train_queue.put(grads)
    train_queue.put(None)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', default='cpu', help='Device to run on default: cpu')
    parser.add_argument('-n', '--name', required=True, help='Name of the run')
    args = parser.parse_args()
    device = torch.device(args.dev)
    
    env = make_env()
    net = common.AtariA2C(env.observation_space.shape, env.action_space.n).to(device)
    net.share_memory()

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    data_proc_list = []
    for proc_idx in range(PROCESSES_COUNT):
        proc_name = f"-a3c-grad_pong_{args.name}-{proc_idx}"
        p_args = (proc_name, net, device, train_queue)
        data_proc = mp.Process(target=grads_func, args=p_args)
        data_proc.start()
        data_proc_list.append(data_proc)

    batch = []
    step_idx = 0
    grad_buffer = None

    try:
        while True:
            train_entry = train_queue.get()
            if train_entry is None:
                break

            step_idx += 1

            if grad_buffer is None:
                grad_buffer = train_entry
            else:
                for tgt_grad, grad in zip(grad_buffer, train_entry):
                    tgt_grad += grad
            
            if step_idx % TRAIN_BATCH == 0:
                optimizer.zero_grad()
                for param, grad in zip(net.parameters(), grad_buffer):
                    param.grad = torch.FloatTensor(grad).to(device)

                nn.utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()
                grad_buffer = None
    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()


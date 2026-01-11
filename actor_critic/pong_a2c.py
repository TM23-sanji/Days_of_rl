from rl.policy_gradients.cartpole_pg import entropy_loss_t
from rl.policy_gradients.cartpole_pg import logits_t
from numpy._core._asarray import require
import gymnasium as gym, ptan, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from ptan.experience import VectorExperienceSourceFirstLast
from ptan.common.utils import TBMeanTracker
import numpy as np, argparse
from torch.utils.tensorboard import SummaryWriter
from lib import common

"""
exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

for step_idx, exp in enumerate(exp_source):

exp : 
ExperienceFirstLast(
    state=S0, 
    action=Buy, 
    reward=1.81,  # Calculated as: (1 * 0.9^0) + (0 * 0.9^1) + (1 * 0.9^2) = 1 + 0 + 0.81
    last_state=S3 # This is the state AFTER 3 steps
)

batch = [
    # exp 0: Episode continues
    ExperienceFirstLast(state=S0, action=0, reward=1.81, last_state=S3),
    
    # exp 1: Episode continues
    ExperienceFirstLast(state=S1, action=1, reward=0.9,  last_state=S4),
    
    # exp 2: EPISODE ENDS at step 2!
    ExperienceFirstLast(state=S2, action=0, reward=1.0,  last_state=None) 
]
"""

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
NUM_ENVS = 50

REWARD_STEPS = 4
CLIP_GRAD = 0.1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', default='cpu', help='Device to run on default: cpu')
    parser.add_argument('--use-async', default=False, action='store_true', help='Use async vector env A3C model')   
    parser.add_argument('-n', "--name", required=True, help='Name of the run')
    args = parser.parse_args()
    device = torch.device(args.dev)

    env_factories = [
        lambda : ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))
        for _ in range(NUM_ENVS)
    ]

    if args.use_async:
        env = gym.vector.AsyncVectorEnv(env_factories)
    else:
        env = gym.vector.SyncVectorEnv(env_factories)
    
    writer = SummaryWriter(comment="-pong-a2c_" + args.name)

    net = common.AtariA2C(env.observation_space.shape, env.action_space.n).to(device)
    print(net)

    agent = ptan.agent.PolicyAgent(
        lambda x: net(x)[0],
        apply_softmax=True,
        device=device
    )

    exp_source = VectorExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    batch = []

    with common.RewardTracker(writer, stop_reward=18) as tracker:
        with TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)

                # handle new rewards
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    if tracker.reward(new_rewards[0], step_idx):
                        break

                if len(batch) < BATCH_SIZE:
                    continue

                # we need common unpack batches as Gamma ^ n * Value of last state this comes from net

                states_t, actions_t, rewards_t = common.unpack_batch(batch, net, device, GAMMA, REWARD_STEPS)
                batch.clear()

                optimizer.zero_grad()
                logits_t, value_t = net(states_t)
                loss_value_t = F.mse_loss(value_t.squeeze(-1), rewards_t)

                log_prob_t = F.log_softmax(logits_t, dim=1)
                adv_t = rewards_t - value_t.detach()
                log_act_t = log_prob_t[range(BATCH_SIZE), actions_t]
                log_prob_action_t = adv_t * log_act_t
                loss_policy_t = -log_prob_action_t.mean()

                prob_t = F.softmax(logits_t, dim=1)
                entropy_loss_t = ENTROPY_BETA * (prob_t * log_prob_t).sum(dim=1).mean()

                # calculate policy gradients only
                loss_policy_t.backward(retain_graph=True)
                grads = np.concatenate([
                    p.grad.data.cpu().numpy().flatten()
                    for p in net.parameters() if p.grad is not None
                ])

                # apply entropy and value gradients
                loss_v = entropy_loss_t + loss_value_t
                loss_v.backward()
                nn.utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()
                loss_v += loss_policy_t

                tb_tracker.track("advantage", adv_t, step_idx)
                tb_tracker.track("values", value_t, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_t, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_t, step_idx)
                tb_tracker.track("loss_policy", loss_policy_t, step_idx)
                tb_tracker.track("loss_value", loss_value_t, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)
                tb_tracker.track("grad_l2", np.sqrt(np.mean(np.square(grads))), step_idx)
                tb_tracker.track("grad_max", np.max(np.abs(grads)), step_idx)
                tb_tracker.track("grad_var", np.var(grads), step_idx)
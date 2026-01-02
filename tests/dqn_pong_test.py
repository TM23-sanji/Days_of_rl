import pytest, gymnasium as gym, numpy as np, torch
from stable_baselines3.common import atari_wrappers
from model.dqn_model import DQN
from envs.wrappers import wrapper 

# https://x.com/i/grok?conversation=1995908053187920196

@pytest.fixture
def base_env():
    return gym.make('PongNoFrameskip-v4')

def test_image_to_pytorch(base_env):
    wrapped = wrapper.ImageToPyTorch(base_env)
    obs, _ = wrapped.reset()
    assert obs.shape == (3, 210, 160)  # Channels first: (C, H, W)
    assert obs.dtype == np.uint8
    # Simulate step
    action = wrapped.action_space.sample()
    next_obs, _, _, _, _ = wrapped.step(action)
    assert next_obs.shape == (3, 210, 160)

def test_buffer_wrapper(base_env):
    wrapped = wrapper.BufferWrapper(wrapper.ImageToPyTorch(base_env), n_steps=4)
    obs, _ = wrapped.reset()
    assert obs.shape == (4, 3, 210, 160)  # Stacked
    assert np.all(obs[:3] == base_env.observation_space.low)  # Padded with low
    # Step 5x: Buffer rolls, shape stable
    for _ in range(5):
        action = wrapped.action_space.sample()
        next_obs, _, _, _, _ = wrapped.step(action)
        assert next_obs.shape == (4, 3, 210, 160)
    # Change n_steps=2: Shape updates to (2,3,210,160)
    new_wrapped = wrapper.BufferWrapper(wrapper.ImageToPyTorch(base_env), n_steps=2)
    new_obs, _ = new_wrapped.reset()
    assert new_obs.shape == (2, 3, 210, 160)

def test_full_env_make():
    env = wrapper.make_env('PongNoFrameskip-v4')
    assert env.observation_space.shape == (4, 84, 84)  # After WarpFrame + stack (grayscale C=1, but stacked=4)
    obs, _ = env.reset()
    assert obs.shape == (4, 84, 84)
    # Quick episode sim: Check rewards in [-1,0,1] if clipped
    total_reward = 0
    for _ in range(100):  # Short sim
        action = env.action_space.sample()
        obs, rew, term, trunc, _ = env.step(action)
        assert obs.shape == (4, 84, 84)
        total_reward += rew
        if term or trunc:
            break
    assert -21 <= total_reward <= 21  # Pong bounds

# For DQN integration
def test_dqn_forward(env):
    net = DQN(env.observation_space.shape, env.action_space.n)
    obs = torch.ByteTensor(np.random.randint(0, 256, (1, *env.observation_space.shape), dtype=np.uint8))
    q_out = net(obs)
    assert q_out.shape == (1, env.action_space.n)

print('Everything good nig chill dude :)')
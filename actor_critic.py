import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matlab.engine
import numpy as np
import compute_reward

# From here: https://github.com/yc930401/Actor-Critic-pytorch/blob/master/Actor-Critic.py
# Another good implementation that uses a nn with a shared first layer:
# https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
import compute_reward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v0").unwrapped

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
lr = 0.0001

eng = matlab.engine.start_matlab()
eng.cd(r'T1D_VPP/', nargout=0)

# Discritize the action space
basal_arr = np.arange(0.01, 3.01, 0.01)  # [0.01, 3] increments of 0.01
carb_arr = np.arange(5, 21, 1)  # [5, 20] increments of 1
num_actions = len(basal_arr) * len(carb_arr)

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value


def compute_returns(next_value, rewards,gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R
        returns.insert(0, R)
    return returns


def environment(action=3069):
    # default action is basal = 0.7, carb = 15
    if action > num_actions or action < 0:
        raise Exception(f"Action must be between: [0, {num_actions}]: {action}")
    basal = basal_arr[action % len(basal_arr)]
    carb = carb_arr[action // len(basal_arr)]

    times = [102, 144, 216, 240]
    carbs = [27, 17, 28, 12]

    m_times = matlab.double(times)
    m_carbs = matlab.double(carbs)

    bg, insulin = eng.run_sim(m_times, m_carbs, carb, basal, nargout=2)
    bg, insulin = bg[0], insulin[0]
    reward = compute_reward.reward(bg)

    state = [basal, carb]
    state.extend(bg)

    return np.array(state), reward


def trainIters(actor, critic, n_iters):
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    for iter in range(n_iters):
        state, _ = environment()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        env.reset()

        for i in count():
            state = torch.FloatTensor(state).to(device)
            probs, value = actor(state), critic(state)

            action = probs.sample()
            next_state, reward = environment(action)

            log_prob = probs.log_prob(action).unsqueeze(0)
            entropy += probs.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))

            state = next_state


        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()
    torch.save(actor, 'model/actor.pkl')
    torch.save(critic, 'model/critic.pkl')
    env.close()


if __name__ == '__main__':
    print(device)
    if os.path.exists('model/actor.pkl'):
        actor = torch.load('model/actor.pkl')
        print('Actor Model loaded')
    else:
        actor = Actor(state_size, action_size).to(device)
    if os.path.exists('model/critic.pkl'):
        critic = torch.load('model/critic.pkl')
        print('Critic Model loaded')
    else:
        critic = Critic(state_size, action_size).to(device)
    trainIters(actor, critic, n_iters=100)
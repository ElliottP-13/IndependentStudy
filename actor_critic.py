import collections
import os

import matplotlib.pyplot as plt
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eng = matlab.engine.start_matlab()
eng.cd(r'T1D_VPP/', nargout=0)

# Discritize the action space
lr = 0.0001

gbest_reward = float('-inf')
gbest_action = -1


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, self.action_size)

    def forward(self, state):
        output = self.linear1(state)
        probs = F.softmax(output, dim=-1)
        print(f'probs: {probs.cpu().detach().numpy()}')
        distribution = Categorical(probs)
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 1)

    def forward(self, state):
        output = self.linear1(state)
        return output


class environment:
    def __init__(self, basal, carb, basal_inc, carb_inc):
        self.basal = basal
        self.carb = carb
        self.basal_inc = basal_inc
        self.carb_inc = carb_inc

    def get_state(self, action=8):
        # default action is basal = 0.7, carb = 15
        if action > self.get_action_size() or action < 0:
            raise Exception(f"Action must be between: [0, {self.get_action_size()}): {action}")

        basal, carb = self.update_insulin(action)
        self.basal, self.carb = basal, carb  # update globals

        # basal = basal_arr[action % len(basal_arr)]
        # carb = carb_arr[action // len(basal_arr)]

        times = [102, 144, 216, 240]
        carbs = [27, 17, 28, 12]

        m_times = matlab.double(times)
        m_carbs = matlab.double(carbs)

        bg, insulin = eng.run_sim(m_times, m_carbs, float(carb), float(basal), nargout=2)
        bg, insulin = bg[0], insulin[0]
        reward = compute_reward.reward(bg)

        state = [basal, carb]
        state.extend(bg)

        print(f"Current State: {basal}, {carb} --> {reward}")
        return np.array(state), reward

    def update_insulin(self, action):
        carb, basal = self.carb, self.basal
        inc_carb, inc_basal = self.carb_inc, self.basal_inc

        f_basal = action % 3
        f_carb = action // 3

        if f_basal == 0:
            basal -= inc_basal
        elif f_basal == 2:
            basal += inc_basal

        if f_carb == 0:
            carb -= inc_carb
        elif f_carb == 2:
            carb += inc_carb

        # bounds checking
        if carb <= 0:
            carb = inc_carb
        if basal <= 0:
            basal = inc_basal

        return basal, carb

    @staticmethod
    def get_state_size():
        return 291

    @staticmethod
    def get_action_size():
        return 9


def compute_returns(next_value, rewards, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R
        returns.insert(0, R)
    return returns


def trainIters(actor, critic, n_iters):
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    env = environment(0.7, 20, 0.3, 5)
    state, _ = env.get_state()

    global gbest_reward, gbest_action

    for iter in range(n_iters):
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0

        for i in range(3):
            state = torch.FloatTensor(state).to(device)
            probs, value = actor(state), critic(state)

            show_distrib(probs, iter)
            print(f'Predicted reward: {value.item()}')

            action = probs.sample()
            next_state, reward = env.get_state(action)

            if reward >= gbest_reward:
                f = open('log.txt', 'a')
                f.write(f"Best action: {action}: ({env.basal}, {env.carb}) = {reward}\n")
                f.close()
                gbest_reward = reward
                gbest_action = action

            log_prob = probs.log_prob(action).unsqueeze(0)
            entropy += probs.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))

            state = next_state

        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards)

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


def show_distrib(distrib, iter):
    print('showing distrib', end=', ')
    x = [i for i in range(environment.get_action_size())]
    n = 10000
    samp = [distrib.sample().item() for _ in range(n)]
    print('finished sampling', end=', ')
    counter = collections.Counter(samp)
    print('made counter', end=', ')
    y = [counter[i] for i in range(environment.get_action_size())]
    print('displaying')

    bin_size = 20
    bin_x = [i for i in range(0, environment.get_action_size(), bin_size)]
    bin_y = [sum(y[bin_x[i]:bin_x[i+1]]) for i in range(len(bin_x) - 1)]

    plt.plot(x, y)
    plt.title(f"Iteration: {iter}")
    plt.show()



if __name__ == '__main__':
    print(device)

    actor = Actor(environment.get_state_size(), environment.get_action_size()).to(device)
    critic = Critic(environment.get_state_size(), environment.get_action_size()).to(device)
    trainIters(actor, critic, n_iters=100)

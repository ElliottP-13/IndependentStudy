import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque

from matplotlib import pyplot as plt

from environment import Environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class Learner(nn.Module):
    def __int__(self):
        super(Learner, self).__int__()


    def select_action(self, state, steps_done):
        # parameters for selecting best action vs random action
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 200

        sample = random.random()
        # compute threshold (by temp)
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                print('thingy')
                o = self(state)
                action = o.argmax().cpu().item()
                return action
        else:
            return random.randrange(self.action_size)

    def add_trace(self, state, action):
        if (state, action) in self.traces:
            self.traces[(state, action)] += 1
        else:
            self.traces[(state, action)] = 1



class SARSA(Learner):
    def __init__(self, state_size, action_size):
        super(SARSA, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 500)
        self.linear2 = nn.Linear(500, 500)
        self.linear3 = nn.Linear(500, self.action_size)

        self.traces = {}

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)

        return output


class CNN_SARSA(Learner):
    def __init__(self, state_size, action_size, other_size=2):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.other_size = other_size

        self.conv1 = nn.Conv1d(1, 16, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(32)

        # conv_out_size = ((state_size - other_size) - 3) // 2 + 1  # not working right, manually set 8994
        self.linear1 = nn.Linear(8994, 128)
        self.linear2 = nn.Linear(128, action_size)


    def forward(self, state):
        insulin, bg = torch.hsplit(state, [self.other_size])

        cv = F.relu(self.bn1(self.conv1(bg.unsqueeze(1))))
        cv = F.relu(self.bn2(self.conv2(cv)))

        cv = cv.view(cv.size(0), -1)  # flatten back out
        o = torch.cat([insulin, cv], dim=1)
        o = F.relu(self.linear1(o))
        output = F.relu(self.linear2(o))

        return output


def train_iters(envs, num_episodes, policy_net, optimizer, TARGET_UPDATE=5, fpath='log.txt'):
    if not isinstance(envs, list):
        env = envs
    else:
        env = random.choice(envs)  # get random patient

    # Initialize the environment and state
    state, _, prev_reward = env.get_state()
    state = torch.tensor(state).unsqueeze(0).to(device)  # to tensor, add batch dimension

    f = open(fpath, 'a')
    f.write(f'New Experiment - Num episodes: {num_episodes}\n')
    f.write('iteration, action, basal, carb, reward\n')

    action = policy_net.select_action(state, 0)

    i = 0
    while True:
        if isinstance(envs, list) and random.random() < 0.3:  # 30% chance to switch to new patient
            f.write('switching patient\n')
            env = random.choice(envs)

            # compute all the initial stuff we need
            state, _, prev_reward = env.get_state()
            state = torch.tensor(state).unsqueeze(0).to(device)  # to tensor, add batch dimension

        # Select and perform an action
        next_state, _, reward = env.get_state(action)

        next_state = torch.tensor(next_state).unsqueeze(0).to(device)

        print(f"{i} action: {action}")
        print(f"Current State: {env.basal} {env.carb} --> {reward}")

        reward = torch.tensor([reward], device=device)

        next_action = policy_net.select_action(next_state, i)

        delta = reward + GAMMA * policy_net(next_state)[0][next_action].detach()  # get Q of next state
        policy_net.add_trace(state.clone(), action)

        state_action_value = None
        expected_values = None

        remove_keys = []

        for key, value in policy_net.traces.items():
            if value < 0.01:  # get rid of small traces
                remove_keys.append(key)
                continue
            s, a = key

            sav = policy_net(s).gather(1, torch.tensor([a], device=device).unsqueeze(0))
            state_action_value = torch.vstack((state_action_value, sav)) \
                if state_action_value is not None else sav
            expected_value = LEARNING_RATE * delta * value
            expected_values = torch.vstack((expected_values, expected_value)) \
                if expected_values is not None else expected_value

            policy_net.traces[key] = LAMBDA * value

        for k in remove_keys:
            policy_net.traces.pop(k)

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_value, expected_values)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        # Move to the next state
        state = next_state
        action = next_action
        prev_reward = reward
        i += 1


# global parameters
BATCH_SIZE = 32
GAMMA = 0.25
LAMBDA = 0.8
LEARNING_RATE = 0.1

if __name__ == "__main__":
    # Base experiments

    policy_net = SARSA(Environment.get_state_size(), Environment.get_action_size()).to(device)

    opt = optim.RMSprop(policy_net.parameters())

    env = Environment(0.7, 20, 0.3, 5)

    print('*' * 100)
    print('Base NNN Experiments')
    print('*' * 100)

    train_iters(env, 150, policy_net, opt, fpath='results/base_sarsa.txt')
    torch.save(policy_net, 'model/base_sarsa.pkl')

    # Random experiments
    envs = [Environment(random.uniform(0.1, 1.5), random.randint(10, 30), 0.3, 5, variable_intake=True, patient=(i + 1))
            for i in range(50)]

    policy_net = SARSA(Environment.get_state_size(), Environment.get_action_size()).to(device)

    opt = optim.RMSprop(policy_net.parameters())

    print('*' * 100)
    print('150 Random CNN Experiments')
    print('*' * 100)

    train_iters(envs, 150, policy_net, opt, fpath='results/random_150_sarsa.txt')

    torch.save(policy_net, 'model/random_150_sarsa.pkl')

    policy_net = SARSA(Environment.get_state_size(), Environment.get_action_size()).to(device)

    opt = optim.RMSprop(policy_net.parameters())

    print('*' * 100)
    print('150 Random CNN Experiments')
    print('*' * 100)

    train_iters(envs, 1000, policy_net, opt, fpath='results/random_1000_sarsa.txt')

    torch.save(policy_net, 'model/random_1000_sarsa.pkl')

    pass
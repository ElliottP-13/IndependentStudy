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


def optimize_model(optimizer):
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    # non_final_next_states = torch.vstack([s for s in batch.state if s is not None])  # BATCH_SIZE x STATE
    # state_batch = torch.vstack(batch.state)  # BATCH_SIZE x STATE

    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)

    action_batch = torch.unsqueeze(torch.cat(batch.action), 1)  # BATCH_SIZE x 1
    reward_batch = torch.cat(batch.reward)  # BATCH_SIZE,

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def train_iters(envs, num_episodes, models, optimizer, TARGET_UPDATE=5, fpath='log.txt'):
    if not isinstance(envs, list):
        env = envs
    else:
        env = random.choice(envs)  # get random patient

    policy_net, target_net = models
    # Initialize the environment and state
    state, _, prev_reward = env.get_state()
    prev_reward = torch.tensor([prev_reward], device=device)
    state = torch.tensor(state).unsqueeze(0).to(device)  # to tensor, add batch dimension

    f = open(fpath, 'a')
    f.write(f'New Experiment - Num episodes: {num_episodes}\n')
    f.write('iteration, action, basal, carb, reward\n')

    action = policy_net.select_action(state, 0)

    for i in range(num_episodes):
        if isinstance(envs, list) and random.random() < 0.3:  # 30% chance to switch to new patient
            f.write('switching patient\n')
            env = random.choice(envs)

            # compute all the initial stuff we need
            state, _, prev_reward = env.get_state()
            prev_reward = torch.tensor([prev_reward], device=device)
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

        for key, value in policy_net.traces.items():
            if value < 0.01:  # get rid of small traces
                policy_net.traces.pop(key)
                continue
            s, a = key

            sav = policy_net(s).gather(1, torch.tensor([a]).unsqueeze(0))
            state_action_value = torch.vstack((state_action_value, sav)) \
                if state_action_value is not None else sav
            expected_value = LEARNING_RATE * delta * value
            expected_values = torch.vstack((expected_values, expected_value)) \
                if expected_values is not None else expected_value

            policy_net.traces[key] = LAMBDA * value

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

    print('Complete')
    f.close()

# global parameters
BATCH_SIZE = 16
GAMMA = 0.999
LAMBDA = 0.8
LEARNING_RATE = 1

if __name__ == "__main__":
    # Base experiments

    policy_net = SARSA(Environment.get_state_size(), Environment.get_action_size()).to(device)
    target_net = SARSA(Environment.get_state_size(), Environment.get_action_size()).to(device)

    opt = optim.RMSprop(policy_net.parameters())

    env = Environment(0.7, 20, 0.3, 5)

    print('*' * 100)
    print('Base NNN Experiments')
    print('*' * 100)

    train_iters(env, 150, (policy_net, target_net), opt, fpath='results/base_nnn.txt')

    pass
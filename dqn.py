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


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 500)
        self.linear2 = nn.Linear(500, 500)
        self.linear3 = nn.Linear(500, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)

        return output

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


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

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


def train_iters(env, num_episodes, models, TARGET_UPDATE=5):
    policy_net, target_net = models
    # Initialize the environment and state
    state, _, _ = env.get_state()
    state = torch.tensor(state).to(device)

    f = open('log.txt', 'a')
    f.write(f'New Experiment - Num episodes: {num_episodes}\n')
    f.write('iteration, action, basal, carb, reward\n')

    for i in range(num_episodes):
        # Select and perform an action
        action = policy_net.select_action(state, i)
        next_state, _, reward = env.get_state(action)

        next_state = torch.tensor(next_state).to(device)

        print(f"{i} action: {action}")
        print(f"Current State: {env.basal} {env.carb} --> {reward}")

        reward = torch.tensor([reward], device=device)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Update the target network, copying all weights and biases in DQN
        if i % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        f.write(f'{i}, {action}, {env.basal}, {env.carb}, {reward}\n')

    print('Complete')
    f.close()

# global parameters
BATCH_SIZE = 128
GAMMA = 0.999
memory = ReplayMemory(10000)

if __name__ == "__main__":
    policy_net = DQN(Environment.get_state_size(), Environment.get_action_size()).to(device)
    target_net = DQN(Environment.get_state_size(), Environment.get_action_size()).to(device)

    optimizer = optim.RMSprop(policy_net.parameters())

    env = Environment(0.7, 20, 0.3, 5)
    train_iters(env, 150, (policy_net, target_net))

    pass
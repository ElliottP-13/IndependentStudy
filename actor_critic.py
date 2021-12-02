import collections
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matlab.engine
from environment import Environment

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
        self.linear1 = nn.Linear(self.state_size, 500)
        self.linear2 = nn.Linear(500, 500)
        self.linear3 = nn.Linear(500, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)

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
    env = Environment(0.7, 20, 0.3, 5)
    actor_state, critic_state, _ = env.get_state()

    learn_len = 7

    global gbest_reward, gbest_action

    with open('log.txt', 'a') as f:
        f.write("New experiment: \n")

    for iter in range(n_iters):
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0

        for i in range(learn_len):
            actor_state = torch.FloatTensor(actor_state).to(device)
            critic_state = torch.FloatTensor(critic_state).to(device)
            probs, value = actor(actor_state), critic(critic_state)

            show_distrib(probs, iter)
            print(f'Predicted reward: {value.item()}')

            action = probs.sample()
            print(f"Action: {action}")
            next_actor_state, next_critic_state, reward = env.get_state(action)

            if reward >= gbest_reward:
                f = open('log.txt', 'a')
                f.write(f"Best action: {action}: ({env.basal}, {env.carb}) = {reward}, iteration {iter * learn_len + i}\n")
                f.close()
                gbest_reward = reward
                gbest_action = action

            log_prob = probs.log_prob(action).unsqueeze(0)
            entropy += probs.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))

            actor_state = next_actor_state
            critic_state = next_critic_state

        next_actor_state = torch.FloatTensor(next_actor_state).to(device)
        next_critic_state = torch.FloatTensor(next_critic_state).to(device)

        next_value = critic(next_critic_state)
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
    x = [i for i in range(Environment.get_action_size())]
    n = 10000
    samp = [distrib.sample().item() for _ in range(n)]
    print('finished sampling', end=', ')
    counter = collections.Counter(samp)
    print('made counter', end=', ')
    y = [counter[i] for i in range(Environment.get_action_size())]
    print('displaying')

    bin_size = 20
    bin_x = [i for i in range(0, Environment.get_action_size(), bin_size)]
    bin_y = [sum(y[bin_x[i]:bin_x[i+1]]) for i in range(len(bin_x) - 1)]

    plt.plot(x, y)
    plt.title(f"Iteration: {iter}")
    plt.show()


if __name__ == '__main__':
    print(device)

    actor = Actor(Environment.get_state_size(), Environment.get_action_size()).to(device)
    critic = Critic(Environment.get_state_size(), Environment.get_action_size()).to(device)
    trainIters(actor, critic, n_iters=100)

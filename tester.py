import torch
from environment import Environment
from dqn import *


def eval_model(m, e, state):
    # Select and perform an action
    with torch.no_grad():
        o = m(state)
        action = o.argmax().cpu().item()

    next_state, _, reward = e.get_state(action)

    next_state = torch.tensor(next_state).unsqueeze(0).to(device)

    print(f"action: {action}")
    print(f"Current State: {e.basal} {e.carb} --> {reward}")

    reward = torch.tensor([reward], device=device)

    # Perform one step of the optimization (on the policy network)

    s = f'{action}, {e.basal}, {e.carb}, {reward.cpu().item()}, '
    return next_state, s


def test(models, env, iterations, fpath='log.txt'):
    f = open(fpath, 'a')
    f.write('New Test \n')
    header = [f'{n}_action, {n}_basal, {n}_carb, {n}_reward' for (n, _) in models]
    f.write('iteration, ' + ', '.join(header) + '\n')

    state, _, _ = env.get_state()
    state = torch.tensor(state).unsqueeze(0).to(device)  # to tensor, add batch dimension

    envs = [env.__copy__() for _ in range(len(models))]
    states = [state.clone() for _ in range(len(models))]

    for i in range(iterations):
        f.write(f'{i}, ')
        for idx, (name, model) in enumerate(models):
            print(name)
            e = envs[idx]
            st = states[idx]
            next_st, s = eval_model(model, e, st)
            states[idx] = next_st
            f.write(s)
        f.write('\n')


if __name__ == '__main__':
    m1 = torch.load('./model/random_1000_cnn.pkl')
    m2 = torch.load('./model/random_1000_nnn.pkl')
    om1 = torch.load('./model/original_runs/random_1000_cnn.pkl')
    om2 = torch.load('./model/original_runs/random_1000_nnn.pkl')

    environment = Environment(random.uniform(0.1, 1.5), random.randint(10, 30), 0.3, 5, variable_intake=True, patient=95)

    test([('CNN1', m1), ('NNN1', m2), ('CNN2', om1), ('NNN2', om2)], environment, 100, fpath='results/test95.csv')


    pass

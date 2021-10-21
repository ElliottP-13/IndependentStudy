import bisect
import random

# Reward function from "An Insulin Bolus Advisor for Type 1 Diabetes Using Deep Reinforcement Learning" Zhu et. al.
# Interpret reward function as:
# -2      if g <= 55
# -1.5    if 55 < g <= 80
# 0.5     if 80 < g <= 180
# -0.8    if 180 < g <= 300
# -1      if 300 < g <= 400
# -2      if 400 < g
reward_ranges = [  55,     80,   180,     300,   400]
rewards =       [-2,  -1.5,   0.5,   -0.8,    -1,    -2]


def reward_point(g):
    indx = bisect.bisect_left(reward_ranges, g)
    return rewards[indx]


def reward(glucose_data):
    return sum([reward_point(g) for g in glucose_data])


if __name__ == '__main__':
    trend = [int(random.randint(30, 400)) for _ in range(10000)]
    print(reward(trend))
    pass
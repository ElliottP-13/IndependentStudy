import random

import matlab.engine
import numpy as np
import compute_reward

eng = matlab.engine.start_matlab()
eng.cd(r'T1D_VPP/', nargout=0)


class Environment:
    def __init__(self, basal, carb, basal_inc, carb_inc, patient=1, variable_intake=False):
        self.basal = basal
        self.carb = carb
        self.basal_inc = basal_inc
        self.carb_inc = carb_inc
        self.nn = patient
        self.variable_intake = variable_intake

        self.state_dict = {}

    def store_state(self, state, reward):
        self.state_dict[(self.basal, self.carb)] = (state, reward)

    def get_adjacent(self):
        out = []
        for a in range(self.get_action_size()):
            if a == 4:
                continue
            basal, carb, _ = self.update_insulin(a)
            if (basal, carb) in self.state_dict:
                out.append((a, self.state_dict[(basal,carb)]))
        return out

    def get_state(self, action=8):
        # default action is basal = 0.7, carb = 15
        if action > self.get_action_size() or action < 0:
            raise Exception(f"Action must be between: [0, {self.get_action_size()}): {action}")

        basal, carb, deltas = self.update_insulin(action)
        self.basal, self.carb = basal, carb  # update globals

        # basal = basal_arr[action % len(basal_arr)]
        # carb = carb_arr[action // len(basal_arr)]

        times = [102, 144, 216, 240]
        carbs = [27, 17, 28, 12]

        if self.variable_intake:  # randomize carb intake a little bit
            times = [x + random.randrange(-30, 30, 1) for x in times]
            carbs = [x + random.randrange(-10, 20, 1) for x in carbs]

        m_times = matlab.double(times)
        m_carbs = matlab.double(carbs)

        bg, insulin = eng.run_sim(m_times, m_carbs, float(carb), float(basal), self.nn, nargout=2)
        bg, insulin = bg[0], insulin[0]
        reward = compute_reward.reward(bg)

        bg = np.array(bg)

        norm_bg = bg / 150
        norm_carb = carb / 15

        critic_state = np.array([deltas[0], deltas[1]])  # delta basal, delta carb
        critic_state = np.append(critic_state, norm_bg)

        actor_state = np.array([basal, norm_carb])
        actor_state = np.append(actor_state, norm_bg)

        print(f"Current State: {basal}, {carb} --> {reward}")
        return np.array(actor_state, dtype=np.float32), np.array(critic_state), reward

    def update_insulin(self, action):
        carb, basal = self.carb, self.basal
        inc_carb, inc_basal = self.carb_inc, self.basal_inc

        f_basal = action % 3
        f_carb = action // 3

        db = 0
        dc = 0

        if f_basal == 0:
            basal -= inc_basal
            db = -inc_basal
        elif f_basal == 2:
            basal += inc_basal
            db = inc_basal

        if f_carb == 0:
            carb -= inc_carb
            dc = -inc_carb
        elif f_carb == 2:
            carb += inc_carb
            dc = inc_carb

        # bounds checking
        if carb <= 0:
            carb = inc_carb
        elif carb >= 50:
            carb = 50

        if basal <= 0:
            basal = inc_basal
        elif basal >= 5:
            basal = 5

        return basal, carb, (db, dc)

    @staticmethod
    def get_state_size():
        return 291

    @staticmethod
    def get_action_size():
        return 9

    def __copy__(self):
        return Environment(self.basal, self.carb, self.basal_inc, self.carb_inc, self.nn, self.variable_intake)

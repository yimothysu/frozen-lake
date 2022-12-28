from random import random, randint

import gym
import gym.envs.toy_text.frozen_lake
import numpy as np


class QTable:
    def __init__(self, size, alpha=0.5, gamma=0.99, epsilon=0.99):
        self.size = size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q = np.zeros((size * size, 4), dtype=np.float64)

    def update(self, state, action, next_state, reward, is_terminal=False):
        s = state
        a = action
        s_ = next_state
        r = -1 if (is_terminal and reward <= 0) else reward
        print(s, a, r)
        alpha = self.alpha

        target = r + int(not is_terminal) * self.gamma * np.amax(self.Q[s_, :])
        self.Q[s, a] = (1 - alpha) * self.Q[s, a] + alpha * target

    def run_policy(self, state):
        s = state
        argmax = np.random.choice(np.flatnonzero(self.Q[s, :] == self.Q[s, :].max()))

        if random() < self.epsilon:
            return randint(0, 3)
        else:
            return argmax

    def __repr__(self):
        return str(self.Q)


def epsilon(iteration, step_count):
    return max(0, 1 - (iteration / step_count))


NUM_STEPS = 1000
SIZE = 4

if __name__ == "__main__":
    if SIZE not in (4, 8):
        print(f"Invalid map size {SIZE}. Must be 4 or 8.")
        exit()

    env = gym.make(
        "FrozenLake-v1",
        desc=gym.envs.toy_text.frozen_lake.generate_random_map(size=SIZE),
        map_name=f"{SIZE}x{SIZE}",
        is_slippery=False,
        render_mode="human",
    )
    observation, info = env.reset(seed=42)

    q_table = QTable(SIZE)

    for i in range(NUM_STEPS):
        if i % 100 == 0:
            print(f"Iteration {i}")
            print("---")
            print(q_table)
            print("---")
            print()

        action = q_table.run_policy(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        is_terminal = terminated or truncated

        q_table.update(
            state=observation,
            action=action,
            next_state=next_observation,
            reward=reward,
            is_terminal=is_terminal,
        )
        q_table.epsilon = epsilon(i, NUM_STEPS / 2)

        if is_terminal:
            next_observation, info = env.reset()
        observation = next_observation

    env.close()

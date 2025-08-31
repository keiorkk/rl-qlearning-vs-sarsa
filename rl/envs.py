import numpy as np
from dataclasses import dataclass

A_UP, A_RIGHT, A_DOWN, A_LEFT = 0, 1, 2, 3
ACTIONS = [A_UP, A_RIGHT, A_DOWN, A_LEFT]
ACTION_DELTAS = {
    A_UP:    (-1, 0),
    A_RIGHT: (0,  1),
    A_DOWN:  (1,  0),
    A_LEFT:  (0, -1),
}

def to_state(r, c, ncols): return r * ncols + c

@dataclass
class GridWorld:
    rows: int = 5
    cols: int = 5
    start: tuple = (0, 0)
    goal: tuple = (4, 4)
    step_reward: float = -1.0
    goal_reward: float = 10.0
    max_steps: int = 200

    def __post_init__(self):
        self.nS = self.rows * self.cols
        self.nA = 4
        self.reset()

    def reset(self):
        self.r, self.c = self.start
        self.steps = 0
        return to_state(self.r, self.c, self.cols)

    def step(self, action: int):
        dr, dc = ACTION_DELTAS[action]
        nr, nc = self.r + dr, self.c + dc
        # stay if hit boundary
        nr = min(max(nr, 0), self.rows - 1)
        nc = min(max(nc, 0), self.cols - 1)
        self.r, self.c = nr, nc
        self.steps += 1

        done = (self.r, self.c) == self.goal or self.steps >= self.max_steps
        reward = self.goal_reward if (self.r, self.c) == self.goal else self.step_reward
        return to_state(self.r, self.c, self.cols), reward, done, {}

@dataclass
class CliffWalking:
    # Classic 4x12 cliff: start (3,0), goal (3,11), cliff (3,1..10)
    rows: int = 4
    cols: int = 12
    start: tuple = (3, 0)
    goal: tuple = (3, 11)
    step_reward: float = -1.0
    cliff_reward: float = -100.0
    max_steps: int = 1000

    def __post_init__(self):
        self.nS = self.rows * self.cols
        self.nA = 4
        self._build_cliff()
        self.reset()

    def _build_cliff(self):
        self.cliff = set((3, c) for c in range(1, self.cols - 1))

    def reset(self):
        self.r, self.c = self.start
        self.steps = 0
        return to_state(self.r, self.c, self.cols)

    def step(self, action: int):
        dr, dc = ACTION_DELTAS[action]
        nr, nc = self.r + dr, self.c + dc
        nr = min(max(nr, 0), self.rows - 1)
        nc = min(max(nc, 0), self.cols - 1)
        self.r, self.c = nr, nc
        self.steps += 1

        done = (self.r, self.c) == self.goal or self.steps >= self.max_steps
        if (self.r, self.c) in self.cliff:
            # fall off cliff -> reset to start, heavy penalty, episode continues (as in classic)
            reward = self.cliff_reward
            self.r, self.c = self.start
            s = to_state(self.r, self.c, self.cols)
            done = False
        else:
            reward = self.step_reward
            s = to_state(self.r, self.c, self.cols)
        if (self.r, self.c) == self.goal:
            reward = 0.0  # in many definitions goal yields 0 with termination
            done = True
        return s, reward, done, {}

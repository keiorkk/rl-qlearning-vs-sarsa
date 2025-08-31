import numpy as np
from .envs import ACTIONS

class TabularAgentBase:
    def __init__(self, n_states: int, n_actions: int, alpha=0.1, gamma=0.99, eps=1.0):
        self.nS, self.nA = n_states, n_actions
        self.alpha, self.gamma = alpha, gamma
        self.eps = eps
        self.Q = np.zeros((n_states, n_actions), dtype=float)

    def act(self, s: int) -> int:
        if np.random.rand() < self.eps:
            return np.random.randint(self.nA)
        return int(np.argmax(self.Q[s]))

class QLearningAgent(TabularAgentBase):
    def update(self, s, a, r, s_next, done):
        best_next = np.max(self.Q[s_next]) if not done else 0.0
        td_target = r + self.gamma * best_next
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

class SARSAAgent(TabularAgentBase):
    def update(self, s, a, r, s_next, a_next, done):
        next_q = self.Q[s_next, a_next] if not done else 0.0
        td_target = r + self.gamma * next_q
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

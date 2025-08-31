import argparse, os, time, csv
import numpy as np
import matplotlib.pyplot as plt
from .envs import GridWorld, CliffWalking, ACTIONS
from .agents import QLearningAgent, SARSAAgent
from .utils import set_seed, linear_epsilon, moving_average

def make_env(name: str):
    if name == "gridworld":
        return GridWorld()
    elif name == "cliff":
        return CliffWalking()
    else:
        raise ValueError(f"Unknown env: {name}")

def train_algo(env_name: str, algo: str, episodes: int, alpha: float, gamma: float,
               eps_start: float, eps_end: float, seed: int = 42):
    env = make_env(env_name)
    set_seed(seed)

    if algo == "qlearning":
        agent = QLearningAgent(env.nS, 4, alpha=alpha, gamma=gamma, eps=eps_start)
    elif algo == "sarsa":
        agent = SARSAAgent(env.nS, 4, alpha=alpha, gamma=gamma, eps=eps_start)
    else:
        raise ValueError("algo must be one of: qlearning, sarsa")

    eps_schedule = list(linear_epsilon(eps_start, eps_end, episodes))
    episode_returns = []

    for ep in range(episodes):
        agent.eps = eps_schedule[ep]
        s = env.reset()
        a = agent.act(s) if algo == "sarsa" else None
        total_r = 0.0
        steps = 0

        done = False
        while not done:
            if algo == "sarsa":
                # on-policy: action already chosen
                s_next, r, done, _ = env.step(a)
                a_next = agent.act(s_next)
                agent.update(s, a, r, s_next, a_next, done)
                s, a = s_next, a_next
            else:
                # q-learning: off-policy
                a = agent.act(s)
                s_next, r, done, _ = env.step(a)
                agent.update(s, a, r, s_next, done)
                s = s_next

            total_r += r
            steps += 1
            if steps > 10000:
                break

        episode_returns.append(total_r)

    return episode_returns

def save_csv(path, returns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "return"])
        for i, r in enumerate(returns, 1):
            w.writerow([i, r])

def plot_curves(curves_dict, window=50, title="", out_path=None):
    plt.figure(figsize=(8, 5))
    for label, series in curves_dict.items():
        ma = moving_average(series, window=window)
        plt.plot(ma, label=f"{label} (MA{window})", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
    else:
        plt.show()
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, default="gridworld", choices=["gridworld", "cliff"])
    ap.add_argument("--algo", type=str, default="qlearning", help="qlearning,sarsa or comma-separated to compare")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ma-window", type=int, default=50)
    args = ap.parse_args()

    algos = [a.strip().lower() for a in args.algo.split(",")]
    curves = {}
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    for algo in algos:
        returns = train_algo(args.env, algo, args.episodes, args.alpha, args.gamma, args.eps_start, args.eps_end, args.seed)
        curves[algo.upper()] = returns
        save_csv(f"results/logs/{args.env}_{algo}_{timestamp}.csv", returns)

    title = f"{args.env.upper()} — {', '.join(a.upper() for a in algos)}"
    out = f"results/curves/{args.env}_{'-'.join(algos)}_{timestamp}.png"
    plot_curves(curves, window=args.ma_window, title=title, out_path=out)
    print(f"Saved curve: {out}")

if __name__ == "__main__":
    main()

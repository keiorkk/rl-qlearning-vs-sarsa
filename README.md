# Reinforcement Learning — Q-Learning vs SARSA

Comparison of Q-Learning and SARSA on custom GridWorld environments.  
Implemented as tabular (table-based) methods, with  `numpy` and `matplotlib` dependencies.

# Features
- Pure Python, self-implemented `GridWorld` / `CliffWalking` (no gym required).
- Both `Q-Learning` and `SARSA` with a unified training loop.
- Linear epsilon decay (`1.0 → 0.05`), default 1000 episodes.
- Automatic learning curve plotting (episode returns + moving average), supports **side-by-side comparison

## Quickstart
```bash
# 1) Create virtual environment (optional)
python -m venv .venv && source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run experiments (compare both algorithms on the same plot)
bash experiments/run_gridworld.sh
bash experiments/run_cliff.sh

# Single algorithm
python rl/train.py --env gridworld --algo qlearning --episodes 1000

# Compare both algorithms 
python rl/train.py --env cliff --algo qlearning,sarsa --episodes 1000


## Structure
```
.
├─ README.md
├─ requirements.txt
├─ rl/
│  ├─ envs.py        # 自定义环境（GridWorld, CliffWalking）
│  ├─ agents.py      # QLearningAgent, SARSAAgent
│  ├─ train.py       # 训练主脚本（支持多算法同图对比）
│  └─ utils.py       # ε衰减、滑动平均、随机种子
├─ experiments/
│  ├─ run_gridworld.sh
│  └─ run_cliff.sh
└─ results/
   ├─ curves/        # 输出图片
   └─ logs/          # 回报CSV
```

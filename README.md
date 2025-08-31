# 强化学习 —— Q-Learning & SARSA 实装对比

对比 Q-Learning 与 SARSA 在 **自定义 GridWorld** 与 **CliffWalking** 两个环境下的学习曲线表现。  
实现为 **表格型（Tabular）**，依赖仅 `numpy`、`matplotlib`。

## 特性
- 纯 Python，自实现 `GridWorld` / `CliffWalking` 环境（无需 gym）。
- `Q-Learning` 与 `SARSA` 两种算法，统一训练循环。
- 线性 `ε` 衰减（从 1.0 → 0.05），默认 1000 回合。
- 自动绘制学习曲线（每回合回报 + 滑动平均），支持**同时对比两条曲线**。

## 快速开始
```bash
# 1) 建虚拟环境（可选）
python -m venv .venv && source .venv/bin/activate

# 2) 安装依赖
pip install -r requirements.txt

# 3) 运行实验（两种算法同图对比）
bash experiments/run_gridworld.sh
bash experiments/run_cliff.sh
```

或手动运行：
```bash
# 单独跑一种算法
python rl/train.py --env gridworld --algo qlearning --episodes 1000

# 同图对比两种算法（用逗号分隔）
python rl/train.py --env cliff --algo qlearning,sarsa --episodes 1000
```

## 目录结构
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

## 超参数（默认值）
- 学习率 `α=0.1`，折扣 `γ=0.99`，回合数 `1000`
- ε 衰减：从 `1.0` 线性下降到 `0.05`（贯穿全部回合）
- 行为空间：上/右/下/左（0/1/2/3）

> 注：在 `CliffWalking` 中，`Q-Learning` 通常更激进（探索靠近悬崖，可能掉落，损失大但收敛路径短），`SARSA` 更保守（倾向远离悬崖，回报更稳定）。不同随机种子下曲线略有差异，属于正常现象。

## 许可证
MIT

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 找到项目根目录和三次实验的日志位置
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = (
    PROJECT_ROOT
    / "official-source"
    / "on-policy-main"
    / "onpolicy"
    / "scripts"
    / "results"
    / "MPE"
    / "simple_spread"
    / "rmappo"
)

TAG = "average_episode_rewards"
SEEDS = [1, 2, 3]


def load_reward_curve(seed):
    """读取一个随机种子的奖励曲线。"""
    log_dir = (
            RESULTS_ROOT
            / f"mpe_spread_rmappo_parallel8_100k_seed{seed}"
            / "run1"
            / "logs"
            / "average_episode_rewards"
            / "average_episode_rewards"
    )
    event_reader = EventAccumulator(str(log_dir))
    event_reader.Reload()
    events = event_reader.Scalars(TAG)

    if not events:
        raise RuntimeError(f"没有找到 seed={seed} 的奖励日志：{log_dir}")

    steps = np.array([event.step for event in events])
    rewards = np.array([event.value for event in events])
    return steps, rewards


# 读取三条曲线
curves = [load_reward_curve(seed) for seed in SEEDS]

# 将三次实验对齐到共同的训练步数
last_common_step = min(steps[-1] for steps, _ in curves)
common_steps = curves[0][0]
common_steps = common_steps[common_steps <= last_common_step]

aligned_rewards = np.vstack(
    [np.interp(common_steps, steps, rewards) for steps, rewards in curves]
)

# 计算平均奖励和标准差（波动范围）
mean_reward = aligned_rewards.mean(axis=0)
std_reward = aligned_rewards.std(axis=0)

# 保存图片与数据
figure_dir = PROJECT_ROOT / "figures"
figure_dir.mkdir(exist_ok=True)

figure_path = figure_dir / "rmappo_parallel8_100k_3seeds.png"
csv_path = figure_dir / "rmappo_parallel8_100k_3seeds.csv"

plt.figure(figsize=(8, 5))
plt.plot(common_steps, mean_reward, color="tab:blue", label="Mean reward")
plt.fill_between(
    common_steps,
    mean_reward - std_reward,
    mean_reward + std_reward,
    color="tab:blue",
    alpha=0.2,
    label="± 1 standard deviation",
)
plt.xlabel("Environment steps")
plt.ylabel("Average team reward (higher is better)")
plt.title("RMAPPO on MPE simple_spread: 3 random seeds")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(figure_path, dpi=200)
plt.close()

np.savetxt(
    csv_path,
    np.column_stack((common_steps, mean_reward, std_reward)),
    delimiter=",",
    header="environment_steps,mean_reward,std_reward",
    comments="",
)

print(f"图片已保存：{figure_path}")
print(f"数据已保存：{csv_path}")
print(f"最后一步平均奖励：{mean_reward[-1]:.2f} ± {std_reward[-1]:.2f}")
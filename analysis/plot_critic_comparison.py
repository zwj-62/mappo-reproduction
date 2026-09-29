from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "figures"

centralized = np.loadtxt(
    FIGURE_DIR / "rmappo_parallel8_100k_3seeds.csv",
    delimiter=",",
    skiprows=1,
)

local_critic = np.loadtxt(
    FIGURE_DIR / "localcritic_parallel8_100k_3seeds.csv",
    delimiter=",",
    skiprows=1,
)

plt.figure(figsize=(8, 5))

for data, label, color in [
    (centralized, "Centralized critic (RMAPPO)", "tab:blue"),
    (local_critic, "Local critic control", "tab:orange"),
]:
    steps, mean_reward, std_reward = data.T
    plt.plot(steps, mean_reward, label=label, color=color)
    plt.fill_between(
        steps,
        mean_reward - std_reward,
        mean_reward + std_reward,
        color=color,
        alpha=0.18,
    )

plt.xlabel("Environment steps")
plt.ylabel("Average team reward (higher is better)")
plt.title("Centralized vs. local critic on MPE simple_spread")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

output_path = FIGURE_DIR / "centralized_vs_localcritic_3seeds.png"
plt.savefig(output_path, dpi=200)
plt.close()

print(f"对比图已保存：{output_path}")
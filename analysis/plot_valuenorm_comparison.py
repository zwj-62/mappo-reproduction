"""ValueNorm 开/关对比：读取已完成的三种子日志，只绘图，不启动训练。

开启组使用本次复跑的 run2，关闭组使用 run1；不覆盖旧的 Critic 对比实验。
以后重跑会产生新 run，需要明确修改 RUN_NUMBERS，避免自动混入其他训练。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = (ROOT / "official-source" / "on-policy-main" / "onpolicy"
                / "scripts" / "results" / "MPE" / "simple_spread" / "rmappo")
# 明确选择每个种子的哪一次训练。on 的 run1 是以前的实验，保留不动。
RUN_NUMBERS = {"on": {1: 2, 2: 2, 3: 2}, "off": {1: 1, 2: 1, 3: 1}}
REWARD_SUFFIX = "/average_episode_rewards/average_episode_rewards"


def run_directory(group, seed):
    suffix = "" if group == "on" else "_novaluenorm"
    name = f"mpe_spread_rmappo_parallel8_100k{suffix}_seed{seed}"
    return RESULTS_ROOT / name / f"run{RUN_NUMBERS[group][seed]}"


def load_record(group, seed):
    run_dir = run_directory(group, seed)
    record = json.loads((run_dir / "reproduction_record.json").read_text(encoding="utf-8"))
    if (record["status"] != "completed" or record["completed_training_steps"] != 100000
            or record["group"] != group or record["seed"] != seed):
        raise RuntimeError(f"Incomplete or mislabeled formal experiment: {run_dir}")
    # 按项目相对位置读取，不依赖迁移前的绝对路径。
    record["run_dir"] = str(run_dir)
    return record


def load_curve(run_dir):
    data = json.loads((Path(run_dir) / "logs" / "summary.json").read_text(encoding="utf-8"))
    keys = [key for key in data if ("/" + key.replace("\\", "/")).endswith(REWARD_SUFFIX)]
    if len(keys) != 1:
        raise ValueError(f"Expected exactly one training-reward metric, found {keys}")
    events = np.asarray(data[keys[0]], dtype=float)
    if events.ndim != 2 or events.shape[1] != 3 or len(events) == 0 or not np.isfinite(events).all():
        raise ValueError("Reward log is empty, malformed, or non-finite")
    steps = events[:, 1]
    if not np.equal(steps, np.floor(steps)).all() or not (np.diff(steps) > 0).all():
        raise ValueError("Expected strictly increasing integer environment-step labels")
    return steps.astype(np.int64), events[:, 2]


def align_curves(curves):
    if not curves:
        raise ValueError("No curves")
    common = curves[0][0]
    for steps, _ in curves[1:]:
        common = np.intersect1d(common, steps)
    if len(common) == 0:
        raise ValueError("No exact shared logging steps")
    aligned = np.vstack([values[np.searchsorted(steps, common)] for steps, values in curves])
    if not np.isfinite(aligned).all():
        raise ValueError("Non-finite aligned rewards")
    return common, aligned


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = []
    curves = []
    for group in ("on", "off"):
        for seed in (1, 2, 3):
            record = load_record(group, seed)
            records.append(record)
            curves.append(load_curve(record["run_dir"]))
    if len({record["source_sha256"] for record in records}) != 1:
        raise RuntimeError("Different source snapshots")
    reference = records[0]["resolved_config"]
    allowed_differences = {"seed", "experiment_name", "use_valuenorm"}
    for record in records:
        config = record["resolved_config"]
        differences = {key for key in reference.keys() | config.keys()
                       if reference.get(key) != config.get(key)}
        if differences - allowed_differences:
            raise RuntimeError(f"Uncontrolled settings: {differences}")
        if record["resolved_config"]["use_valuenorm"] != (record["group"] == "on"):
            raise RuntimeError("ValueNorm group mislabeled")
    steps, values = align_curves(curves)
    if not np.array_equal(steps, np.arange(200, 100000, 1000)):
        raise RuntimeError("Formal curves do not cover the expected common budget")
    output = ROOT / "figures"
    output.mkdir(exist_ok=True)
    rows = []
    summary = {
        "seeds": [1, 2, 3], "budget_per_run": 100000,
        "last_logged_step": int(steps[-1]), "logged_points": len(steps),
        "metric": "Training proxy: mean(buffer.rewards) * episode_length; not independent evaluation",
        "std_ddof": 0, "last_window": "last 10 common logged points, 90200..99200",
        "source_sha256": records[0]["source_sha256"], "groups": {},
    }
    prefix_checks = {}
    for group, curve_index in (("on", 0), ("off", 3)):
        smoke_suffix = "" if group == "on" else "_novaluenorm"
        smoke_name = f"mpe_spread_rmappo_parallel8_10k{smoke_suffix}_seed1"
        smoke_run = 2 if group == "on" else 1
        smoke_steps, smoke_values = load_curve(RESULTS_ROOT / smoke_name / f"run{smoke_run}")
        if not np.array_equal(smoke_steps, steps[:len(smoke_steps)]):
            raise RuntimeError("Smoke/formal step labels differ")
        prefix_checks[group] = float(np.max(np.abs(smoke_values - values[curve_index, :len(smoke_steps)])))
    summary["smoke_vs_formal_seed1_prefix_max_abs_difference"] = prefix_checks
    summary["on_off_initial_logged_reward_difference_per_seed"] = (values[:3, 0] - values[3:, 0]).tolist()
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for index, (group, label, color) in enumerate((
        ("on", "ValueNorm on", "tab:red"), ("off", "ValueNorm off", "tab:blue")
    )):
        group_values = values[index * 3:(index + 1) * 3]
        mean, std = group_values.mean(axis=0), group_values.std(axis=0, ddof=0)
        window_means = group_values[:, -10:].mean(axis=1)
        summary["groups"][group] = {
            "last_point_mean": float(mean[-1]), "last_point_std": float(std[-1]),
            "last_10_mean": float(window_means.mean()),
            "last_10_std_across_seed_means": float(window_means.std(ddof=0)),
            "per_seed_last_point": group_values[:, -1].tolist(),
            "per_seed_last_10_mean": window_means.tolist(),
            "elapsed_seconds": [record["elapsed_seconds"] for record in records[index * 3:(index + 1) * 3]],
        }
        axis.plot(steps, mean, label=label, color=color, linewidth=1.7)
        axis.fill_between(steps, mean - std, mean + std, color=color, alpha=0.18)
        for seed, curve, record in zip((1, 2, 3), group_values, records[index * 3:(index + 1) * 3]):
            rows.append([group, seed, 100000, int(steps[-1]), float(curve[-1]), float(curve[-10:].mean()), record["elapsed_seconds"]])
        np.savetxt(output / f"valuenorm_{group}_3seeds.csv",
                   np.column_stack((steps, group_values.T, mean, std)), delimiter=",",
                   header="environment_steps,seed1,seed2,seed3,mean_reward,std_reward_ddof0", comments="")
    axis.set(xlabel="Environment steps", ylabel="Training return proxy (higher is better)",
             title="ValueNorm ablation: RMAPPO / MPE simple_spread\n3 seeds; shading = +/- 1 cross-seed standard deviation")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "valuenorm_on_vs_off_3seeds.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for index, (group, axis) in enumerate(zip(("on", "off"), axes)):
        for seed, curve in zip((1, 2, 3), values[index * 3:(index + 1) * 3]):
            axis.plot(steps, curve, label=f"Seed {seed}", alpha=0.8, linewidth=1)
        axis.set(title=f"ValueNorm {group}", xlabel="Environment steps")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Training return proxy (higher is better)")
    figure.tight_layout()
    figure.savefig(output / "valuenorm_individual_seeds.png", dpi=180)
    plt.close(figure)
    with (output / "valuenorm_per_seed_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["valuenorm", "seed", "completed_steps", "last_logged_step", "last_point_reward", "last_10_logged_points_mean", "elapsed_seconds"])
        writer.writerows(rows)
    paired_difference = values[:3, -10:].mean(axis=1) - values[3:, -10:].mean(axis=1)
    summary["on_minus_off_last_10_paired_seed_differences"] = paired_difference.tolist()
    summary["on_minus_off_last_10_mean_difference"] = float(paired_difference.mean())
    (output / "valuenorm_training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for group, label in (("on", "开启 ValueNorm"), ("off", "关闭 ValueNorm")):
        stats = summary["groups"][group]
        print(f"{label}: 最后日志点 {stats['last_point_mean']:.2f} ± {stats['last_point_std']:.2f}; "
              f"末 10 点 {stats['last_10_mean']:.2f} ± {stats['last_10_std_across_seed_means']:.2f}")
    print(f"对比图片：{output / 'valuenorm_on_vs_off_3seeds.png'}")
    print(f"逐种子数据：{output / 'valuenorm_per_seed_summary.csv'}")


if __name__ == "__main__":
    main()

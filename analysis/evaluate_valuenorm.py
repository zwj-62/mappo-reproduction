"""可选的独立评估：读取保存的 Actor，使用共同的新初始状态，不训练。

与训练曲线使用相同的六个 run（见 plot_valuenorm_comparison.py）。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from plot_valuenorm_comparison import ROOT, RESULTS_ROOT, load_record

sys.path.insert(0, str(ROOT / "official-source" / "on-policy-main"))
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.envs.mpe.MPE_env import MPEEnv


@torch.no_grad()
def evaluate_actor(record, episodes, device):
    args = SimpleNamespace(**record["resolved_config"])
    env = MPEEnv(args)
    try:
        actor = R_Actor(args, env.observation_space[0], env.action_space[0], device=device)
        actor.load_state_dict(torch.load(Path(record["run_dir"]) / "models" / "actor.pt", map_location=device, weights_only=True), strict=True)
        actor.eval()
        assert all(space.__class__.__name__ == "Discrete" for space in env.action_space)
        rows = []
        initial_hashes = []
        for episode in range(episodes):
            evaluation_seed = 100000 + episode
            env.seed(evaluation_seed)
            obs = np.asarray(env.reset(), dtype=np.float32)
            initial_hashes.append(hashlib.sha256(obs.tobytes()).hexdigest())
            states = np.zeros((args.num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
            masks = np.ones((args.num_agents, 1), dtype=np.float32)
            agent_returns = np.zeros(args.num_agents, dtype=np.float64)
            for step in range(args.episode_length):
                actions, _, hidden = actor(obs, states, masks, deterministic=True)
                action_indices = actions.cpu().numpy().reshape(args.num_agents)
                actions_env = np.stack([np.eye(space.n)[int(action)] for space, action in zip(env.action_space, action_indices)])
                obs, rewards, dones, _ = env.step(actions_env)
                obs = np.asarray(obs, dtype=np.float32)
                agent_returns += np.asarray(rewards, dtype=float).reshape(args.num_agents)
                states = hidden.cpu().numpy()
                done = np.asarray(dones, dtype=bool)
                if step < args.episode_length - 1 and done.any():
                    raise RuntimeError("Unexpected early termination")
                if step == args.episode_length - 1 and not done.all():
                    raise RuntimeError("Environment did not end at the configured episode length")
            reward = float(agent_returns.mean())
            if not np.isfinite(reward):
                raise RuntimeError("Non-finite evaluation return")
            rows.append([record["group"], record["seed"], evaluation_seed, args.episode_length, reward])
        return rows, initial_hashes
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    options = parser.parse_args()
    if options.episodes < 3:
        parser.error("At least three episodes required for the reproducibility check")
    if options.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(options.device)
    budget = 100000 if options.phase == "formal" else 10000
    seeds = (1, 2, 3) if options.phase == "formal" else (1,)
    all_rows, per_model = [], []
    reference_hashes = None
    source_hash = None
    records = []
    for group in ("on", "off"):
        for seed in seeds:
            if options.phase == "formal":
                record = load_record(group, seed)
            else:
                suffix = "" if group == "on" else "_novaluenorm"
                run = 2 if group == "on" else 1
                run_dir = RESULTS_ROOT / f"mpe_spread_rmappo_parallel8_10k{suffix}_seed1" / f"run{run}"
                record = json.loads((run_dir / "reproduction_record.json").read_text(encoding="utf-8"))
                record["run_dir"] = str(run_dir)
            name = record["name"]
            if record["status"] != "completed" or record["group"] != group or record["seed"] != seed:
                raise RuntimeError(f"Missing/inconsistent trained model: {name}")
            if source_hash is not None and source_hash != record["source_sha256"]:
                raise RuntimeError("Models have different source snapshots")
            source_hash = record["source_sha256"]
            records.append(record)
            rows, hashes = evaluate_actor(record, options.episodes, device)
            if reference_hashes is not None and hashes != reference_hashes:
                raise RuntimeError("Evaluation models did not receive identical initial states")
            reference_hashes = hashes
            all_rows.extend(rows)
            returns = np.asarray([row[-1] for row in rows])
            per_model.append({"group": group, "training_seed": seed,
                              "episode_mean": float(returns.mean()), "episode_std_ddof0": float(returns.std(ddof=0))})
            print(f"EVAL {name}: {returns.mean():.3f} across {options.episodes} held-out episodes", flush=True)
    repeated, repeat_hashes = evaluate_actor(records[0], 3, device)
    repeated_difference = float(np.max(np.abs(np.asarray([row[-1] for row in repeated]) - np.asarray([row[-1] for row in all_rows[:3]]))))
    if repeated_difference > 1e-8 or repeat_hashes != reference_hashes[:3]:
        raise RuntimeError("Deterministic evaluation failed repeatability check")
    summary = {
        "phase": options.phase, "training_budget_per_model": budget,
        "episodes_per_model": options.episodes,
        "evaluation_environment_seeds": [100000, 100000 + options.episodes - 1],
        "deterministic_actions": True, "actor_only": True,
        "device": str(device), "source_sha256": source_hash,
        "identical_initial_states_verified": True, "initial_state_sha256": reference_hashes,
        "repeated_first_3_max_abs_difference": repeated_difference,
        "metric": "Independent held-out episode return, summed over 25 steps and averaged over agents",
        "std_definition": "ddof=0 across the trained-seed episode means; not confidence intervals",
        "per_model": per_model, "groups": {},
    }
    for group in ("on", "off"):
        means = np.array([model["episode_mean"] for model in per_model if model["group"] == group])
        summary["groups"][group] = {"mean": float(means.mean()), "std_across_training_seeds": float(means.std(ddof=0)), "per_seed_means": means.tolist()}
    summary["on_minus_off_mean"] = summary["groups"]["on"]["mean"] - summary["groups"]["off"]["mean"]
    output = ROOT / "figures"
    output.mkdir(exist_ok=True)
    prefix = "valuenorm" if options.phase == "formal" else "valuenorm_smoke"
    with (output / f"{prefix}_evaluation_episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["valuenorm", "training_seed", "evaluation_environment_seed", "episode_length", "episode_return"])
        writer.writerows(all_rows)
    (output / f"{prefix}_evaluation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"groups": summary["groups"], "repeatability_max_abs_difference": repeated_difference}, indent=2))
    print(f"Saved independent evaluation: {output}")


if __name__ == "__main__":
    main()

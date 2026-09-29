# MPE simple_spread：3 个智能体的 RMAPPO 小型实验
# 目标：跑通一个可重复的 10,000 步训练，不等同于论文的完整复现。

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$TrainScript = Join-Path $ProjectRoot "official-source\on-policy-main\onpolicy\scripts\train\train_mpe.py"
$WorkingDir = Join-Path $ProjectRoot "official-source\on-policy-main\onpolicy\scripts"

Set-Location $WorkingDir

& $PythonExe $TrainScript `
  --env_name MPE `
  --algorithm_name rmappo `
  --experiment_name mpe_spread_rmappo_parallel8_100k_seed2 `
  --scenario_name simple_spread `
  --num_agents 3 `
  --num_landmarks 3 `
  --seed 2 `
  --use_wandb `
  --n_training_threads 1 `
  --n_rollout_threads 8 `
  --n_eval_rollout_threads 1 `
  --num_mini_batch 1 `
  --episode_length 25 `
  --num_env_steps 100000 `
  --ppo_epoch 10 `
  --use_ReLU `
  --gain 0.01 `
  --lr 7e-4 `
  --critic_lr 7e-4
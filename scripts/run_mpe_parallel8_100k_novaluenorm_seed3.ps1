# ValueNorm 对照实验：关闭价值归一化，种子 3，100,000 环境步。
# 与 run_mpe_parallel8_100k_seed3.ps1 保持相同设置，仅改变实验名及 ValueNorm 开关。
# 注意：官方 --use_valuenorm 使用 store_false，写上该参数表示关闭，而不是开启。

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$TrainScript = Join-Path $ProjectRoot "official-source\on-policy-main\onpolicy\scripts\train\train_mpe.py"
$WorkingDir = Join-Path $ProjectRoot "official-source\on-policy-main\onpolicy\scripts"

Push-Location $WorkingDir
try {
  & $PythonExe $TrainScript `
    --env_name MPE `
    --algorithm_name rmappo `
    --experiment_name mpe_spread_rmappo_parallel8_100k_novaluenorm_seed3 `
    --scenario_name simple_spread `
    --num_agents 3 `
    --num_landmarks 3 `
    --seed 3 `
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
    --critic_lr 7e-4 `
    --use_valuenorm
  if ($LASTEXITCODE -ne 0) { throw "Training failed: exit code $LASTEXITCODE" }
}
finally {
  Pop-Location
}

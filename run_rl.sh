# # Replay to get rlpd Datasets
# python -m mani_skill.trajectory.replay_trajectory \
#   --traj-path /home/zhouxunzhe/robo_dev/ManiSkill/demos/FoldSuitcase-v1/motionplanning/suitcase_1k.pd_joint_pos.cpu.h5 \
#   --use-first-env-state -c pd_joint_delta_pos -o state --allow-failure --save-traj --num-procs 1

# # Replay to get rfcl Datasets
# python -m mani_skill.trajectory.replay_trajectory  \
#   --traj-path /home/zhouxunzhe/robo_dev/ManiSkill/demos/FoldSuitcase-v1/motionplanning/rfcl/suitcase_1k.pd_joint_pos.cpu.h5 \
#   --use-first-env-state -c pd_joint_delta_pos -o state --save-traj --num-procs 1 --count 100

# Run rlpd Training
env_id=FoldSuitcase-v1
demos=1000
seed=42
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m examples.baselines.rlpd.train_ms3 examples/baselines/rlpd/configs/base_rlpd_ms3.yml \
  logger.exp_name="rlpd-${env_id}-state-${demos}_rl_demos-${seed}-walltime_efficient" logger.wandb=True \
  seed=${seed} train.num_demos=${demos} train.steps=2000000 \
  env.env_id=${env_id} \
  train.dataset_path="/home/zhouxunzhe/robo_dev/ManiSkill/demos/FoldSuitcase-v1/motionplanning/suitcase_1k.state.pd_joint_delta_pos.cpu.h5"

# Run rlpd Training
# XLA_PYTHON_CLIENT_PREALLOCATE=false python -m examples.baselines.rlpd.train_ms3 examples/baselines/rlpd/configs/base_rlpd_ms3.yml \
#   logger.exp_name="rlpd-FoldSuitcase-v1-state-1000_rl_demos-42-walltime_efficient" logger.wandb=True \
#   seed=42 train.num_demos=1000 train.steps=2000000 \
#   env.env_id=FoldSuitcase-v1 \
#   train.dataset_path="/home/zhouxunzhe/robo_dev/ManiSkill/demos/FoldSuitcase-v1/motionplanning/suitcase_1k.state.pd_joint_delta_pos.cpu.h5"

# # Run rfcl Training
# env_id=FoldSuitcase-v1
# demos=1000 # number of demos to train on
# seed=42
# XLA_PYTHON_CLIENT_PREALLOCATE=false python -m examples.baselines.rfcl.train examples/baselines/rfcl/configs/base_sac_ms3_sample_efficient.yml \
#   logger.exp_name=rfcl-${env_id}-state-${demos}_motionplanning_demos-${seed}-walltime_efficient logger.wandb=True \
#   seed=${seed} train.num_demos=${demos} train.steps=2000000 \
#   env.env_id=${env_id} \
#   train.dataset_path="/home/zhouxunzhe/robo_dev/ManiSkill/demos/FoldSuitcase-v1/motionplanning/suitcase_1k.state.pd_joint_delta_pos.cpu.h5"

# # Run PPO Training
# python -m examples.baselines.ppo.ppo --env_id="FoldSuitcase-v1" \
#   --num_envs=1 --update_epochs=8 --num_minibatches=32 \
#   --total_timesteps=2_000_000 --eval_freq=100 --num-steps=500
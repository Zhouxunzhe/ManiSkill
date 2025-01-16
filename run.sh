# # Motion planning to collect pd_joint_pos data
# python -m mani_skill.examples.motionplanning.panda.run \
#   --traj-name "suitcase_1k.pd_joint_pos.cpu" -e "FoldSuitcase-v1" -n 1000 \
#   --obs-mode rgb --only-count-success -b cpu --shader rt --num-procs 1

# # Replay data to get delta pos
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path /home/zhouxunzhe/robo_dev/ManiSkill/demos/FoldSuitcase-v1/motionplanning/suitcase_1k.pd_joint_pos.cpu.h5 \
  --use-first-env-state -c pd_joint_delta_pos -o rgb   --save-traj --num-procs 1

# Train diffusion policy
python -m examples.baselines.diffusion_policy.train_rgbd --env-id FoldSuitcase-v1 \
  --demo-path /home/zhouxunzhe/robo_dev/ManiSkill/demos/FoldSuitcase-v1/motionplanning/suitcase_1k.rgb.pd_joint_delta_pos.cpu.h5 \
  --control-mode "pd_joint_delta_pos" --sim-backend "cpu" --num-demos 400 --max_episode_steps 500 --total_iters 300000 \
  --log_freq 1000 --save_freq 1000 --num_eval_episodes 100 --num_eval_envs 1 --visual_encoder clip --exp_name Fold-v1-CLIP-1K

# # Replay to get RL Datasets
# python -m mani_skill.trajectory.replay_trajectory \
#   --traj-path /home/zhouxunzhe/robo_dev/ManiSkill/demos/FoldSuitcase-v1/motionplanning/suitcase_1k.pd_joint_pos.cpu.h5 \
#   --use-first-env-state -c pd_joint_delta_pos -o state --allow-failure --save-traj --num-procs 1
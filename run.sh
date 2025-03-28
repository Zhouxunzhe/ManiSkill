# # Motion planning to collect pd_joint_pos data
python -m mani_skill.examples.motionplanning.panda.run \
  --traj-name "train_fold_all_500.rgbd.pd_joint_pos.cpu" -e "FoldSuitcase-v1" -n 500 \
  --obs-mode rgbd --only-count-success -b cpu --shader rt --num-procs 1 --data-mode train

# Collect laptop only data
python -m mani_skill.examples.motionplanning.panda.run \
  --traj-name "train_fold_laptop_1k.pd_joint_pos.cpu" -e "FoldSuitcase-v1" -n 1000 \
  --obs-mode rgbd --only-count-success -b cpu --shader rt --num-procs 1 --data-mode laptop

# Collect pick cube ycb data
python -m mani_skill.examples.motionplanning.panda.run \
  --traj-name "multi_task.rgbd.pd_joint_pos.cpu" -e "PickCubeYCB-v1" -n 100 \
  --obs-mode rgbd --only-count-success -b cpu --shader rt --num-procs 1

# Replay data to get delta pos
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path /home/engram/zhouxunzhe/ManiSkill/demos/FoldSuitcase-v1/motionplanning/train_fold_all_500.rgbd.pd_joint_pos.cpu.h5 \
  --use-first-env-state -c pd_joint_delta_pos -o rgbd   --save-traj

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path /home/engram/zhouxunzhe/ManiSkill/demos/PickCubeYCB-v1/motionplanning/multi_task_500.rgbd.pd_joint_pos.cpu.h5 \
  --use-first-env-state -c pd_joint_delta_pos -o rgbd   --save-traj

# # Replay to get RL Datasets
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path /home/engram/zhouxunzhe/ManiSkill/demos/FoldSuitcase-v1/motionplanning/train_fold_all_1k.rgbd.pd_joint_delta_pos.physx_cpu.h5\
  --use-first-env-state -c pd_joint_delta_pos -o state --allow-failure --save-traj --num-procs 1

# Train diffusion policy
python -m examples.baselines.diffusion_policy.train_rgbd --env-id PickCubeYCB-v1 \
  --demo-path /home/engram/zhouxunzhe/ManiSkill/demos/PickCubeYCB-v1/motionplanning/multi_task_500.rgbd.pd_joint_delta_pos.physx_cpu.h5 \
  --control-mode "pd_joint_delta_pos" --shader rt --num-demos 500 --max_episode_steps 500 --total_iters 60000 --batch_size 256 \
  --log_freq 3000 --eval_freq 3000 --save_freq 3000 --num_eval_episodes 100 --num_eval_envs 1 --visual_encoder "plain_conv" \
  --obs_mode rgb+depth --exp_name PickCubeYCB-multi_task-CNN-500

python -m examples.baselines.diffusion_policy.train_rgbd --env-id PickCubeYCB-v1 \
  --demo-path /home/engram/zhouxunzhe/ManiSkill/demos/PickCubeYCB-v1/motionplanning/multi_task_500.rgbd.pd_joint_delta_pos.physx_cpu.h5 \
  --control-mode "pd_joint_delta_pos" --shader rt --num-demos 500 --max_episode_steps 500 --total_iters 60000 --batch_size 256 \
  --log_freq 3000 --eval_freq 3000 --save_freq 3000 --num_eval_episodes 100 --num_eval_envs 1 --visual_encoder "siglip" \
  --obs_mode rgb+depth --exp_name PickCubeYCB-multi_task-Siglip-500

# Train diffusion policy with language condition
python -m examples.baselines.diffusion_policy.train_rgbd_lan --env-id "FoldSuitcase-v1" \
  --demo-path "/home/engram/zhouxunzhe/ManiSkill/demos/FoldSuitcase-v1/motionplanning/train_fold_all_500.rgbd.pd_joint_delta_pos.physx_cpu.h5" \
  --control-mode "pd_joint_delta_pos" --shader rt --num-demos 500 --max_episode_steps 500 --total_iters 60000 --batch_size 256 \
  --log_freq 3000 --eval_freq 3000 --save_freq 3000 --num_eval_episodes 100 --num_eval_envs 1 --visual_encoder "siglip" \
  --obs_mode "rgb+depth" --lan_encoder "encoder_only" --language_condition_type "concat" --sparse_steps 4 \
  --exp_name "Fold-all-CNN-encoder_only-concat-500" --prompt "Fold laptop, box, suitcase"

python -m examples.baselines.diffusion_policy.train_rgbd_lan --env-id "FoldSuitcase-v1" \
  --demo-path "/home/engram/zhouxunzhe/ManiSkill/demos/FoldSuitcase-v1/motionplanning/train_fold_all_500.rgbd.pd_joint_delta_pos.physx_cpu.h5" \
  --control-mode "pd_joint_delta_pos" --shader rt --num-demos 500 --max_episode_steps 500 --total_iters 60000 --batch_size 256 \
  --log_freq 3000 --eval_freq 3000 --save_freq 3000 --num_eval_episodes 100 --num_eval_envs 1 --visual_encoder "siglip" \
  --obs_mode "rgb+depth" --lan_encoder "encoder_only" --language_condition_type "adapter" --sparse_steps 4 \
  --exp_name "Fold-all-CNN-encoder_only-adapter-500" --prompt "Fold laptop, box, suitcase"

python -m examples.baselines.diffusion_policy.train_rgbd_lan --env-id "FoldSuitcase-v1" \
  --demo-path "/home/engram/zhouxunzhe/ManiSkill/demos/FoldSuitcase-v1/motionplanning/train_fold_all_500.rgbd.pd_joint_delta_pos.physx_cpu.h5" \
  --control-mode "pd_joint_delta_pos" --shader rt --num-demos 500 --max_episode_steps 500 --total_iters 60000 --batch_size 256 \
  --log_freq 3000 --eval_freq 3000 --save_freq 3000 --num_eval_episodes 100 --num_eval_envs 1 --visual_encoder "siglip" \
  --obs_mode "rgb+depth" --lan_encoder "encoder_only" --language_condition_type "sparse_actions" --sparse_steps 4 \
  --exp_name "Fold-all-CNN-encoder_only-sparse_actions-500" --prompt "Fold laptop, box, suitcase"

# Foundation Model
python -m examples.baselines.diffusion_policy.train_rgbd_lan --env-id "PickCubeYCB-v1" \
  --demo-path "/home/engram/zhouxunzhe/ManiSkill/demos/PickCubeYCB-v1/motionplanning/multi_task_500.rgbd.pd_joint_delta_pos.physx_cpu.h5" \
  --control-mode "pd_joint_delta_pos" --shader rt --num-demos 500 --max_episode_steps 500 --total_iters 60000 --batch_size 256 \
  --log_freq 3000 --eval_freq 3000 --save_freq 3000 --num_eval_episodes 100 --num_eval_envs 1 --visual_encoder "shared" \
  --obs_mode "rgb+depth" --lan_encoder "encoder_decoder" --language_condition_type "adapter" --sparse_steps 4 \
  --exp_name "PickCubeYCB-multi_task-shared-encoder_decoder-adapter-500"
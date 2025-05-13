# Motion planning to collect pd_joint_pos data
python -m mani_skill.examples.motionplanning.panda.run \
  --traj-name "single_stack_blue_cube_red_cube.rgbd.pd_joint_pos.cpu" -e "PickCubeYCB-v1" -n 80 \
  --obs-mode rgbd --only-count-success -b cpu --shader rt --num-procs 1

# Replay data to get delta pos
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path /home/engram/zhouxunzhe/ManiSkill/demos/PickCubeYCB-v1/motionplanning/single_stack_blue_cube_red_cube.rgbd.pd_joint_pos.cpu.h5 \
  --use-first-env-state -c pd_joint_delta_pos -o rgbd   --save-traj --shader rt --max-retry 5

# python -m mani_skill.trajectory.replay_trajectory \
#   --traj-path /home/engram/zhouxunzhe/ManiSkill/demos/test/pick_red_cube_plate_stable.rgbd.pd_joint_pos.cpu.h5 \
#   --use-first-env-state -c pd_joint_delta_pos -o rgbd   --save-video

# Train diffusion policy
# python -m examples.baselines.diffusion_policy.train_rgbd --env-id PickCubeYCB-v1 \
#   --demo-path /home/engram/zhouxunzhe/ManiSkill/demos/PickCubeYCB-v1/motionplanning/multi_task_stable.rgbd.pd_joint_delta_pos.physx_cpu.h5 \
#   --video-path /home/engram/zhouxunzhe/ManiSkill/examples/baselines/hyper_net/processed_data \
#   --control-mode "pd_joint_delta_pos" --shader rt --num-demos 100 --max_episode_steps 500 --total_iters 60000 --batch_size 256 \
#   --log_freq 5000 --eval_freq 5000 --save_freq 5000 --num_eval_episodes 100 --num_eval_envs 1 --visual_encoder plain_conv \
#   --obs_mode rgb+depth --exp_name PickCubeYCB-pick_red_cube_plate_stable-CNN-100

# Train HyperNet
python -m examples.baselines.hyper_net.train_hypernet_diffusion --env-id PickCubeYCB-v1 \
  --demo-path /home/engram/zhouxunzhe/ManiSkill/demos/PickCubeYCB-v1/motionplanning/pick_red_cube_plate.rgbd.pd_joint_delta_pos.physx_cpu.h5 \
  --video-path /home/engram/zhouxunzhe/ManiSkill/examples/baselines/hyper_net/processed_data \
  --control-mode "pd_joint_delta_pos" --shader rt --num-demos 80 --max_episode_steps 500 --total_iters 60000 --batch_size 128 \
  --log_freq 5000 --eval_freq 5000 --save_freq 5000 --num_eval_episodes 100 --num_eval_envs 1 \
  --obs_mode rgb+depth --exp_name PickCubeYCB-pick_red_cube_plate-hypernet_diffusion-80 \

# Train MLP
# python -m examples.baselines.hyper_net.train_mlp --env-id PickCubeYCB-v1 \
#   --demo-path /home/engram/zhouxunzhe/ManiSkill/demos/PickCubeYCB-v1/motionplanning/multi_task_stable.rgbd.pd_joint_delta_pos.physx_cpu.h5 \
#   --video-path /home/engram/zhouxunzhe/ManiSkill/examples/baselines/hyper_net/processed_data \
#   --control-mode "pd_joint_delta_pos" --shader rt --num-demos 100 --max_episode_steps 500 --total_iters 60000 --batch_size 128 \
#   --log_freq 5000 --eval_freq 5000 --save_freq 5000 --num_eval_episodes 100 --num_eval_envs 1 \
#   --obs_mode rgb+depth --exp_name PickCubeYCB-pick_red_cube_plate_stable-MLP-100

# python -m examples.baselines.diffusion_policy.train_rgbd --env-id PickCubeYCB-v1 \
#   --demo-path /home/engram/zhouxunzhe/ManiSkill/demos/PickCubeYCB-v1/motionplanning/pick_red_cube_plate_stable.rgbd.pd_joint_delta_pos.physx_cpu.h5 \
#   --control-mode "pd_joint_delta_pos" --shader rt --num-demos 500 --max_episode_steps 500 --total_iters 60000 --batch_size 128 \
#   --log_freq 3000 --eval_freq 3000 --save_freq 3000 --num_eval_episodes 100 --num_eval_envs 1 --visual_encoder siglip \
#   --obs_mode rgb+depth --exp_name PickCubeYCB-pick_red_cube_plate_stable-Siglip-500

# Train diffusion policy with language condition
# python -m examples.baselines.diffusion_policy.train_rgbd_lan --env-id PickCubeYCB-v1 \
#   --demo-path /home/engram/zhouxunzhe/ManiSkill/demos/PickCubeYCB-v1/motionplanning/multi_task_stable.rgbd.pd_joint_delta_pos.physx_cpu.h5 \
#   --control-mode pd_joint_delta_pos --shader rt --num-demos 500 --max_episode_steps 500 --total_iters 60000 --batch_size 128 \
#   --log_freq 3000 --eval_freq 3000 --save_freq 3000 --num_eval_episodes 100 --num_eval_envs 1 --visual_encoder shared \
#   --obs_mode rgb+depth --lan_encoder encoder_decoder --language_condition_type adapter --sparse_steps 4 \
#   --exp_name PickCubeYCB-multi_task_stable-shared-encoder_decoder-adapter-500 --prompt "pick red cube and place on plate."
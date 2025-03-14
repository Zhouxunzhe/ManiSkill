python -m examples.baselines.rdt.eval_sim.eval_rdt_maniskill \
  --pretrained_path /home/zhouxunzhe/robo_dev/ManiSkill/examples/baselines/rdt/eval_sim/rdt_ckpt/mp_rank_00_model_states.pt \
  --env-id FoldSuitcase-v1 --obs-mode rgb --num-traj 100 -b cpu --shader rt --num-procs 1
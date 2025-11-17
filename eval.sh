source "/data/yuwenye/miniconda3/etc/profile.d/conda.sh"; conda activate openpi
export PYTHONPATH=/home/yuwenye/project/openpi:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0

python reward_model/eval_reward.py \
--dataset_path /data/yuwenye/reward_modeling/data/original/1113_kitchen.hdf5 \
--output_path /data/yuwenye/reward_modeling/data/sarm/kitchen_100_reward_eval \
--reward_model_path /data/yuwenye/reward_modeling/output/simple_sarm/reward_model_80.pt \
--vision_encoder_path /data/yuwenye/reward_modeling/dinov2_vitb14_reg4_pretrain.pth \

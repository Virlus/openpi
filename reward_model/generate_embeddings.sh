source "/data/yuwenye/miniconda3/etc/profile.d/conda.sh"; conda activate qwen

export WANDB_API_KEY=956bd553be1193c6fdfce758ad1eded5190eac56
export PYTHONPATH=/home/yuwenye/project/openpi:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python reward_model/generate_embeddings.py \
    --data_path /data/yuwenye/reward_modeling/data/original/1113_kitchen.hdf5 \
    --output_path /data/yuwenye/reward_modeling/data/sarm/1122_kitchen_test_qwen3vl_embeddings.hdf5 \
    --backbone qwen3_vl \
    --prompt "Put the items in the pot." \
    --batch_size 32
    # --dino-ckpt-path /data/yuwenye/reward_modeling/dinov2_vitb14_reg4_pretrain.pth \
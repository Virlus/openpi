# Reward modeling guide

## Scripts for reward modeling training

First, extract embeddings using one of the registered backbones in `reward_model/config.py`.

```
python reward_model/generate_embeddings.py \
    --data_path /path/to/raw/hdf5/dataset \
    --output_path /path/to/save/your/embeddings/buffer \
    --backbone dinov2_minilm \
    --dino-ckpt-path /path/to/dinov2_vitb14_reg4_pretrain.pth \
    --prompt "Put the items in the pot." \
    --batch_size 32
```

For end-to-end VLMs (e.g., `qwen3_vl`), change `--backbone` accordingly. A minimal Qwen3-VL extraction example lives at `third_party/qwen3/extract_vl_embeddings.py`.

Then, annotate stage information on the embedding buffer.

```
python reward_model/annotate_stage.py \
    --data_path /path/to/save/your/embeddings/buffer \
    --stage_annotation_path /path/to/stage/annotation/json/file \
    --num_stages ${number of stages in this task} \
    --visual-embedding-key dino_embeddings
```

Finally, train the reward model (see `train.sh` for a concrete invocation).

## Key components for the reward model

### Multi-stage dataset

Defined in `reward_model/multi_stage_dataset.py`. The dataset now accepts arbitrary `visual_embedding_key` and optional `language_embedding_key`, padding sequences uniformly and exposing the inferred embedding dimensions so the transformer can adapt automatically.

### Reward model

Defined in `reward_model/reward_transformer.py`. The model automatically adapts its projection heads to the provided embedding dimensions and can operate with or without language tokens (e.g., pure VLM embeddings like Qwen3-VL). Use the backbone configs in `reward_model/config.py` to keep the pipeline consistent across embedding generation, training, and evaluation.
MODEL_NAME="/path/base_model_path"
INSTANCE_DIR='dog'
CLASS_DIR="reg_dog"
OUTPUT_DIR="trained_models/ktxl_dog_text"
cfg_file=./default_config.yaml

accelerate launch --config_file ${cfg_file} train_dreambooth_lora.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --class_data_dir=$CLASS_DIR \
    --instance_prompt="ktxl狗" \
    --class_prompt="狗" \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=2e-5 \
    --text_encoder_lr=5e-5 \
    --lr_scheduler="polynomial" \
    --lr_warmup_steps=100 \
    --rank=4 \
    --resolution=1024 \
    --max_train_steps=1000 \
    --checkpointing_steps=200 \
    --num_class_images=100 \
    --center_crop \
    --mixed_precision='fp16' \
    --seed=19980818 \
    --img_repeat_nums=1 \
    --sample_batch_size=2 \
    --gradient_checkpointing \
    --adam_weight_decay=1e-02 \
    --with_prior_preservation  \
    --prior_loss_weight=0.7 \
    --train_text_encoder \

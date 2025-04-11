suffix_str='surrounding_thought'
trial_name=Llama-2-13B-chat${suffix_str}
output_model=./outputs/highD/intention_traj/${trial_name}
# 需要修改到自己的输入目录
if [ ! -d ${output_model} ];then  
    mkdir ${output_model}
fi
cp ./finetune_lora.sh ${output_model}
deepspeed --master_port 29509 --include localhost:0,1,2,3 finetune_lora.py \
    --model_name_or_path ./Llama_models/llama/models/13B-chat/hf \
    --train_files ./datasets/highD/intention_traj/llama_train_${suffix_str}.json \
    --validation_files  ./datasets/highD/intention_traj/llama_val_${suffix_str}.json \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --experiment_id ${trial_name} \
    --use_fast_tokenizer true \
    --output_dir ${output_model} \
    --evaluation_strategy  steps \
    --max_eval_samples 1000 \
    --learning_rate 5e-4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --warmup_steps 600 \
    --load_in_bits 8 \
    --lora_r 64 \
    --lora_alpha 16 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --preprocessing_num_workers 64 \
    --save_steps 50 \
    --eval_steps 50 \
    --save_total_limit 20 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 2048 \
    --report_to wandb \
    --project_name LC_Intention_Trajectory_Prediction \
    --run_name ${trial_name} \
    --overwrite_output_dir \
    --deepspeed ds_config_zero2.json \
    --ignore_data_skip true \
    --bf16 \
    --gradient_checkpointing \
    --bf16_full_eval \
    --ddp_timeout 18000000 \
    | tee -a ${output_model}/train.log
    
# --resume_from_checkpoint ${output_model}/checkpoint-20400 \

type_name=surrounding_thought
trival_name=Llama-2-13B-chat_${type_name}
output_model=./outputs/highD/intention_traj/${trival_name}
python inference.py --base_model_path "./Llama_models/llama/models/7B-chat/hf" \
    --new_model  ${output_model} \
    --val_data_path "./datasets/highD/intention_traj/llama_val_${type_name}.json" \
    --device_id 1 \
    --max_new_tokens  300 \
    --reponse_dir  ${output_model}/response_${type_name}.pkl \
    --batch_size_each_gpu 32

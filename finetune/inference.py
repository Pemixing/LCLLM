import re
import torch
import argparse
import json
from tqdm import tqdm
import bitsandbytes as bnb
from trl import SFTTrainer
from functools import partial
from datasets import load_dataset, Dataset
from typing import Tuple
import datetime
import tempfile

import numpy as np
import transformers
import pandas as pd
import wandb
import pickle
import os

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    AutoPeftModelForCausalLM,
    PeftModel
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    set_seed,
    logging
)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3,4,5"

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, hugg_datasets):
        self.hugg_datasets = hugg_datasets
        
    def __len__(self):
        return len(self.hugg_datasets)
    
    def __getitem__(self, i):
        prompt = re.search(r'\[INST\](.*?)\[\/INST\]', self.hugg_datasets[i]['text'], re.DOTALL).group()
        prompt = prompt.strip()
        return prompt


def inference(args):
    base_model_name = args.base_model_path
    new_model = args.new_model_path
    device_map = {"": args.device_id}
    # device_map = "auto"
    val_data_path = args.val_data_path
    max_seq_length = args.max_new_tokens
    reponse_dir = args.reponse_dir
    
    val_hugg_data = json.load(open(val_data_path, 'r'))
    pipe_dataset = MyDataset(val_hugg_data)
    # print(pipe_dataset[0], len(pipe_dataset))
    # for qa json
    # files = open(val_data_path, 'r', encoding='utf-8')
    # val_hugg_data = []
    # for file in files:
    #     data = json.loads(file)
    #     prompt = data["messages"][0]["content"] + data["messages"][1]["content"]
    #     prompt = prompt.strip()
    #     val_hugg_data.append(prompt)
    # print(val_hugg_data[1])
    # prompt = re.search(r'\[INST\](.*?)\[\/INST\]', val_hugg_data[1]['text'], re.DOTALL).group(1)
    # prompt = re.sub(r'<</SYS>> | <<SYS>>', "", prompt)
    # prompt = prompt.strip()
    # print(prompt)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    ).eval()
    merge_model = PeftModel.from_pretrained(base_model, new_model)
    merge_model = merge_model.merge_and_unload().eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = "[PAD]" # tokenizer.eos_token
    tokenizer.padding_side = "left"

    prediction_pipe = pipeline(task="text-generation", model=merge_model, tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id, do_sample=True, \
                    temperature=0.95, max_new_tokens=max_seq_length, repetition_penalty=1.15, return_full_text=False)
    responses = []
    print(pipe_dataset[0])
    print("##########################")
    num = 0
    for i in range(5):
        for results in tqdm(prediction_pipe(pipe_dataset, batch_size=32), total=len(pipe_dataset)):
            print(num)
            print(results[0]['generated_text'])
            responses.append(results[0]['generated_text'])
            with open(reponse_dir, 'wb') as f:
                pickle.dump(responses, f)
            num += 1
   
#     for i in range(len(val_hugg_data)):
#         print(i)
#         # print(val_hugg_data[i]['text'])
#         prompt = re.search(r'\[INST\](.*?)\[\/INST\]', val_hugg_data[i]['text'], re.DOTALL).group(1)
#         # prompt = re.sub(r'<</SYS>> | <<SYS>>', "", prompt)
#         prompt = prompt.strip()
#         # # for qa json
#         # prompt = val_hugg_data[i]
#         print(prompt)
#         result = prediction_pipe(f"<s>[INST] {prompt} [/INST]")
#         print(result[0]['generated_text'])
#     #     responses.append(result[0]['generated_text'])
#     #     with open(reponse_dir, 'wb') as f:
#     #         pickle.dump(responses, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="/hpc2hdd/JH_DATA/share/xguo796/xguo796_AI_Drive/Llama_models/llama/models/7B-chat/hf")
    parser.add_argument("--val_data_path", type=str, default="/hpc2hdd/home/mpeng060/Workspace/Prediction_LLMs/datasets/highD/4s/llama_val_base.json")
    parser.add_argument("--new_model_path", type=str, default="/hpc2hdd/home/mpeng060/Workspace/Prediction_LLMs/outputs/highD/llama_7B_chat_base_4s")
    parser.add_argument("--reponse_dir", type=str, default="/hpc2hdd/home/mpeng060/Workspace/Prediction_LLMs/response/highD/finetune/test.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--batch_size_each_gpu", type=int, default=32)
    parser.add_argument("--device_id", type=int, default=0)

    args = parser.parse_args()

    # set_seed(args.seed)
    
    import time
    st = time.time()
    inference(args)
    print("Total time: ", time.time() - st)
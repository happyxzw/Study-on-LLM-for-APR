import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import fire
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel,PeftConfig
from prompter import Prompter
from datetime import datetime
from openai import OpenAI

client = OpenAI(base_url="https://api.linyinet.asia/v1",
                             api_key="sk-B6lk0as2o9WjlwnQTcakug5yLNuhn9qFCvOagRCwXIkVb0Dy",
                             timeout=300,
                             max_retries=3)
import re

def create_model_and_tokenizer(model_name_or_path, model_type, load_in_8bit=False):
    quantization_config = None
    if load_in_8bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
    )
    if torch.cuda.is_available():
        print(f"Model device_map=auto, {torch.cuda.device_count()} GPUs")
    # model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path
    )
    if 'deepseekcoder' in model_type:
        tokenizer.pad_token_id = 32018 #"<pad>"
    else:
        tokenizer.pad_token_id = 0 # unk. we want this to be different from the eos token
    tokenizer.padding_side = "right"  
    print(model_type + f' pad token id is {tokenizer.pad_token_id}')
    return model, tokenizer

def prompt_text(origin_prompt, model_type, enhance_methods):
    
    if enhance_methods == "cot":
        # sys_msg = "The following function contains a bug.Think step-by-step to fix the bug. Then, write '// Fixed Function' on a new line followed ONLY by the fixed code."
        sys_msg = "The following function contains a bug. Think step-by-step to understand the bug. Then output ONLY the corrected function. After the function, add // comments explaining your fix."
    else:
        sys_msg = "The following function contains a bug. Output only the complete fixed function with NO explanations, NO markdown, NO comments."
        
    if 'codellama' in model_type.lower():
        return f"<s>[INST] <<SYS>>\n{sys_msg}\n<</SYS>>\n\n{origin_prompt} [/INST]\n"
    elif 'deepseekcoder' in model_type.lower():
        return f"{sys_msg}\n### Instruction:\n{origin_prompt}"
    else:
        return f"{sys_msg}\n\n{origin_prompt}"

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
from apr_utils import create_model_and_tokenizer, prompt_text,client
import re

def build_prompt(problem,bug_type,examples=None):
    if examples:
        with open(f"/home/chenshiping/peft4apr/datasets/prompt/repair_prompt_open_source_one.txt") as f:
            repair_prompt = f.read()   
            origin_prompt = repair_prompt.format(example_bug_1=examples[0]['input'],example_fix_1=examples[0]['output'],bug=problem)
    else:
        with open(f"/home/chenshiping/peft4apr/datasets/prompt/repair_prompt_open_source.txt") as f:
            repair_prompt = f.read()
            origin_prompt = repair_prompt.format(bug=problem,bug_type=bug_type)
                      
    return origin_prompt

def oss_response_filter(output):
    pattern = r'```java(.*?)```'
    matches = re.findall(pattern, output, re.DOTALL)
    if len(matches) == 1:
        liens = matches[0].split('\n')
        for l in liens:
            if '@@' in l or '@Override' in l:
                liens.remove(l)
            
        return '\n'.join(liens)
    else:
        return ""
    
def generate_output(
        #benchmark params
        benchmark_data: str = "",
        benchmark_name: str = "humaneval",
        #output params  
        output_dir: str = "",
        output_file_name: str = "",
        #model params
        model_type: str = "",
        model_name_or_path: str = "/home/survolt/warehouse/llmpeft4apr/models/",
        train_dataset: str = "magicoder",
        #peft model params
        is_peft: bool = False,
        peft_model_weights: str = "/home/survolt/warehouse/llmpeft4apr/models/", 
        enhance_methods: str = "lora",
        examples_path: str = "/home/chenshiping/peft4apr/src/benchmarks/example.json",
        #generation config params
        num_output: int = 10,
        max_new_tokens: int = 256,
        max_seq_len: int = 1200,
):
    #format input and output path
    benchmark_data_path = benchmark_data
    benchmark_json = json.load(open(benchmark_data_path))
    output_data_path = f'{output_dir}{output_file_name}'
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    print(f"==========Generating output of {benchmark_name} benchmark by {model_type} ==========")
    assert (model_name_or_path), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    # create base-model and tokenizer
    if 'gpt' in model_type:
        pass
    else:
        model, tokenizer = create_model_and_tokenizer(model_name_or_path, model_type)
    #merge peft model weights
    if is_peft:
        model = PeftModel.from_pretrained(
            model, 
            peft_model_weights,
            torch_dtype=torch.float16,
            )
    output = json.load(open(benchmark_data_path, 'r'))
    output['model'] = model_type
    output['train_dataset'] = train_dataset
    start_time = time.time()
    all_examples = {}
    if enhance_methods == "few-shot":
        all_examples = json.load(open(examples_path))
    for i, proj in enumerate(output['data']):
        text = output['data'][proj]['input']
        bug_info = benchmark_json["data"].get(proj)
        print(f"bug_info{bug_info}")
        bug_type = None
        if bug_info:
            bug_type = bug_info.get("category")
        examples = []
        if enhance_methods == "few-shot":
            if bug_type and bug_type in all_examples:
                examples = all_examples[bug_type][:1]
            else:
                print(f"[WARN] fallback for {proj}, bug_type={bug_type}")
                examples = sum(list(all_examples.values())[:2], [])[:1]
                
        origin_prompt = build_prompt(text, bug_type, examples if enhance_methods == "few-shot" else None)
        prompt = prompt_text(origin_prompt, model_type, enhance_methods)
           
        print(i + 1, 'generating', proj)
        print("=" * 50)
        print("PROMPT TEXT (first 500 chars):")
        print("=" * 50)
        print(prompt[:500])
        print("=" * 50)
        try:
            if 'gpt' in model_type:
                response = client.chat.completions.create(
                    model=model_type,
                    messages = [
                        {"role": "system", "content": "You are an Automatic Program Repair Tool"}, 
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=512,
                    n=num_output,
                    temperature=0.8,                       
                    top_p=0.95, 
                )
                output_list = [choice.message.content for choice in response.choices]
            else:
                input_ids = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=max_seq_len,
                    padding=False,
                    return_tensors="pt",
                )
                eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
                generated_ids = model.generate(
                    input_ids=input_ids['input_ids'].cuda(),
                    max_new_tokens=512,          # 直接用传入的值，不要加 len(input_ids[0])
                    do_sample=True,                         # 开启采样
                    temperature=0.8,                        # 控制随机性，推荐 0.1~0.7
                    top_p=0.95,                             # 核采样
                    num_return_sequences=num_output,        # 生成 num_output 个候选
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_id,
                )
                output_list = []

                for generated_id in generated_ids:
                    raw_output = tokenizer.decode(generated_id[len(input_ids[0]):],skip_special_tokens=False)

                    clean_output = tokenizer.decode(
                        generated_id[len(input_ids[0]):],
                        skip_special_tokens=True
                    )

                    print("RAW :", repr(raw_output))
                    print("CLEAN:", repr(clean_output))

                    output_list.append(clean_output)
                    #output_list.append(tokenizer.decode(generated_id[len(input_ids[0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False))

        except Exception as e:
            output_list = []
            print(e)
        if 'deepseekcoder' in model_type:
            for i, o in enumerate(output_list):
                end_bucket = o.rfind('}')
                output_list[i] = o[:end_bucket+1]
        output['data'][proj]['output'] = output_list

        json.dump(output, open(output_data_path, 'w'), indent=2)
        # break
    total_time = int(time.time() - start_time)
    output['time'] = total_time
    output['benchmark'] = benchmark_name
    json.dump(output, open(output_data_path, 'w'), indent=2)
    print(f"==========Output written to {output_data_path}==========")

if __name__ == '__main__':
    fire.Fire(generate_output)

    
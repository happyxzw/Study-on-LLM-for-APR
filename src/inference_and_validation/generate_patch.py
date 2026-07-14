import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import fire
import time
import torch
from apr_utils import create_model_and_tokenizer, prompt_text, client

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def build_prompt(problem, bug_type, examples=None, use_bug_type=False):
    prompt_dir = os.path.join(PROJECT_ROOT, "datasets", "prompt")
    if examples:
        with open(os.path.join(prompt_dir, "repair_prompt_open_source_rag.txt"), encoding="utf-8") as f:
            repair_prompt = f.read()
            origin_prompt = repair_prompt.format(example_bug_1=examples[0]['input'],example_fix_1=examples[0]['output'],bug=problem, example_bug_2=examples[1]['input'],example_fix_2=examples[1]['output'])
    else:
        template = "repair_prompt_open_source_type.txt" if use_bug_type else "repair_prompt_open_source.txt"
        with open(os.path.join(prompt_dir, template), encoding="utf-8") as f:
            repair_prompt = f.read()
            origin_prompt = repair_prompt.format(bug=problem, bug_type=bug_type)

    return origin_prompt

    
def generate_output(
        #benchmark params
        benchmark_data: str = "",
        benchmark_name: str = "humaneval",
        #output params  
        output_dir: str = "",
        output_file_name: str = "",
        #model params
        model_type: str = "",
        model_name_or_path: str = "",
        enhance_methods: str = "zero-shot",
        use_bug_type: bool = False,
        examples_path: str = "",
        #generation config params
        num_output: int = 10,
        max_new_tokens: int = 512,
        max_seq_len: int = 1200,
):
    #format input and output path
    with open(benchmark_data, encoding="utf-8") as f:
        output = json.load(f)
    
    output['model'] = model_type
    output['benchmark'] = benchmark_name
    output_data_path = os.path.join(output_dir, output_file_name)
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)

     # create base-model and tokenizer    
    print(f"==========Generating output of {benchmark_name} benchmark by {model_type} ==========")
    if "gpt" not in model_type.lower():
        model, tokenizer = create_model_and_tokenizer(model_name_or_path, model_type)
  
    all_examples = {}
    if examples_path:
        with open(examples_path, encoding="utf-8") as f:
            all_examples = json.load(f)
            
    start_time = time.time()     
    for i, proj in enumerate(output['data']):
        text = output['data'][proj]['input']
        bug_type = output["data"][proj].get("category")
        
        examples = None
        if enhance_methods == "few-shot":
            if bug_type in all_examples:
                examples = all_examples[bug_type][:2]
            else:
                print(f"[WARN] fallback for {proj}, bug_type={bug_type}")
                examples = sum(all_examples.values(), [])[:2]

        origin_prompt = build_prompt(text, bug_type, examples, use_bug_type=use_bug_type)
        prompt = prompt_text(origin_prompt, model_type, enhance_methods)
        print(f"\n[{i+1}] {proj}")
        print("-" * 60)
        print(prompt[:500])
        print("-" * 60)
        
        try:
            if 'gpt' in model_type.lower():
                response = client.chat.completions.create(
                    model=model_type,
                    messages = [
                        {"role": "system", "content": "You are an Automatic Program Repair Tool"}, 
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_new_tokens,
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
                generated_ids = model.generate(
                    input_ids=input_ids['input_ids'].cuda(),
                    max_new_tokens=max_new_tokens,          
                    do_sample=True,                         
                    temperature=0.8,                       
                    top_p=0.95,                            
                    num_return_sequences=num_output,        
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.convert_tokens_to_ids(tokenizer.eos_token),
                )
                
                output_list = []
                for generated_id in generated_ids:
                    raw_output = tokenizer.decode(generated_id[len(input_ids[0]):],skip_special_tokens=False)
                    clean_output = tokenizer.decode(generated_id[len(input_ids[0]):],skip_special_tokens=True, clean_up_tokenization_spaces=False)

                    # print("RAW :", repr(raw_output))
                    # print("CLEAN:", repr(clean_output))
                    output_list.append(clean_output)
                    
        except Exception as e:
            output_list = []
            print(e)
        if 'deepseekcoder' in model_type.lower():
            for i, o in enumerate(output_list):
                end_bucket = o.rfind('}')
                output_list[i] = o[:end_bucket+1]
                
        output['data'][proj]['output'] = output_list

        with open(output_data_path, "w") as f:
            json.dump(output, f, indent=2)
        # break
        
    total_time = int(time.time() - start_time)
    output['time'] = total_time
    with open(output_data_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"==========Output written to {output_data_path}==========")

if __name__ == '__main__':
    fire.Fire(generate_output)
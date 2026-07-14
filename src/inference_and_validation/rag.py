from __future__ import absolute_import, division, print_function
import os
import re
import sys
import time
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
import pickle
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
from apr_utils import create_model_and_tokenizer, prompt_text, client

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_embedding_UniXcoder(args, model, tokenizer, think_dataset):

    if os.path.exists(args.kb_path):
        try:
            with open(args.kb_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        
            for file in think_dataset:
                embedding = existing.get(file, {}).get("embedding_UniXcoder")
                if embedding is not None:
                    think_dataset[file]["embedding_UniXcoder"] = embedding
            print(f"Loaded existing UniXcoder embeddings from {args.kb_path}")
            return
        except (json.JSONDecodeError, KeyError):
            print(f"Existing KB corrupted, regenerating...")

    for file, sample in think_dataset.items():
        if "embedding_UniXcoder" in sample:
            continue
        buggy = sample["buggy"]
        tokenized_code = tokenizer.encode_plus(buggy,max_length=400,truncation=True,return_tensors="pt")
        tokenized_code = {k: v.to(model.device)for k, v in tokenized_code.items()}
        outputs = model(**tokenized_code)
        think_dataset[file]["embedding_UniXcoder"] = (outputs[0][0, 0, :].detach().cpu().numpy().tolist())

    tmp_path = args.kb_path + f".tmp{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(think_dataset, f, indent=4)
    os.replace(tmp_path, args.kb_path)
    print(f"Saved think_dataset with UniXcoder embeddings to {args.kb_path}")
 
def get_embedding_TFIDF(args, think_dataset):

    if os.path.exists(args.kb_path) and os.path.exists(args.vec_path):
        try:
            with open(args.kb_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
                
            for file in think_dataset:
                embedding = existing.get(file, {}).get("embedding_TFIDF")
                if embedding is not None:
                    think_dataset[file]["embedding_TFIDF"] = embedding
            print(f"Loaded existing TF-IDF embeddings from {args.kb_path}")
            return
        except (json.JSONDecodeError, KeyError):
            print(f"Existing KB corrupted, regenerating...")

    files = list(think_dataset.keys())
    corpus = [think_dataset[f]["buggy"] for f in files]

    vectorizer = TfidfVectorizer(
        token_pattern=r"[A-Za-z_][A-Za-z0-9_]*|[^\sA-Za-z0-9_]",
        lowercase=False,
        max_features=5000
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    for i, f in enumerate(files):
        think_dataset[f]["embedding_TFIDF"] = tfidf_matrix[i].toarray()[0].tolist()

    tmp_vec = args.vec_path + f".tmp{os.getpid()}"
    with open(tmp_vec, "wb") as f_out:
        pickle.dump(vectorizer, f_out)
    os.replace(tmp_vec, args.vec_path)

    tmp_path = args.kb_path + f".tmp{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f_out:
        json.dump(think_dataset, f_out, indent=4)
    os.replace(tmp_path, args.kb_path)
    print(f"Saved think_dataset with TF-IDF embeddings to {args.kb_path}")

def get_example_from_clusters(args, think_dataset, model, tokenizer, buggy):

    embeddings = np.asarray([item["embedding_UniXcoder"] for item in valid_items])
    valid_items = [{"file": k, **v} for k, v in think_dataset.items()]#if v["category"] == category
    points = [item["file"] for item in valid_items]
    if len(embeddings) == 0:
        return None
    kmeans = KMeans(n_clusters=min(args.n_example, len(embeddings)),n_init=10,random_state=42)
    labels = kmeans.fit_predict(embeddings) 
    
    tokenized_code = tokenizer.encode_plus(buggy, max_length=400, truncation=True, return_tensors="pt")
    tokenized_code = {k: v.to(model.device)for k, v in tokenized_code.items()}
    buggy_embedding = model(**tokenized_code)[0][0, 0, :].detach().cpu().numpy()
    
    selected_points = []
    for i in range(min(args.n_example, len(np.unique(labels)))):
        cluster_points = np.where(labels == i)[0]
        if len(cluster_points) == 0:
            continue
        similarities = cosine_similarity([buggy_embedding], embeddings[cluster_points])
        selected_points.append(points[cluster_points[np.argmax(similarities)]])
        
    print(f"selected_points: {selected_points}")
    return selected_points

def get_example_topk(args, model, tokenizer, buggy, valid_items, buggy_embedding):

    points = [item["file"] for item in valid_items]
    if args.select == "TopK":
        embeddings = np.asarray([item["embedding_TFIDF"] for item in valid_items]) 
    else:
        embeddings = np.asarray([item["embedding_UniXcoder"] for item in valid_items])
        tokenized_code = tokenizer.encode_plus(buggy,max_length=400, truncation=True, return_tensors="pt")
        tokenized_code = {k: v.to(model.device)for k, v in tokenized_code.items()}
        outputs = model(**tokenized_code)
        buggy_embedding = outputs[0][0,0,:].detach().cpu().numpy()
    
    similarities = cosine_similarity([buggy_embedding], embeddings)[0]
    topk = np.argsort(similarities)[::-1][:args.n_example]
    selected_points = [points[i] for i in topk]

    return selected_points, similarities[topk]

def extract_buggy_snippet(buggy_code, fix_code, location, tokenizer, max_tokens=800, context_lines=15):

    if tokenizer is None:
        full_length = len((buggy_code + fix_code).split())
    else:
        full_length = len(tokenizer.tokenize(buggy_code + fix_code))
    if full_length <= max_tokens:
        return buggy_code, fix_code 

    lines_buggy = buggy_code.split('\n')
    lines_fix = fix_code.split('\n')
    if not location:
        return buggy_code[:max_tokens], fix_code[:max_tokens]
    
    buggy_line = location[0]
    start = max(0, buggy_line - context_lines)
    end = min(len(lines_buggy), buggy_line + context_lines)
    
    snippet_buggy = '\n'.join(lines_buggy[start:end])
    snippet_fix = '\n'.join(lines_fix[start:end])
    return snippet_buggy, snippet_fix

def rag_repair(args, Select_model, Select_tokenizer, think_dataset, inference_files, tokenizer, model, output_data_path, vectorizer):
    prompt_dir = os.path.join(PROJECT_ROOT, "datasets", "prompt")
    if args.enhance_methods == "rag":
        with open(os.path.join(prompt_dir, "repair_prompt_open_source_rag.txt")) as f:
            repair_prompt = f.read()
        with open(os.path.join(prompt_dir, "repair_prompt_open_source.txt")) as f:
            zero_shot_prompt = f.read()
    elif args.enhance_methods == "cot":
        with open(os.path.join(prompt_dir, "repair_prompt_cot.txt")) as f:
            repair_prompt = f.read()

    try:
        with open(output_data_path, "r") as f:
            repair = json.load(f)
    except FileNotFoundError:
        repair = {}

    total = 0
    skip_no_buggy = 0
    skip_example = 0
    skip_length = 0
    used_zero_shot = 0
    all_sim_records = []
    
    for file, bug in inference_files.items():
        total += 1
        buggy_code = bug.get("buggy") or bug.get("input", "")
        if not buggy_code:
            print(f"Warning: No buggy code found for {file}")
            skip_no_buggy += 1
            continue
        if args.enhance_methods == "rag":
            valid_items = [{"file": k, **v} for k, v in think_dataset.items()]  
            if not valid_items:
                print(f"Warning: No valid items found in {file}")
                continue
            print("Repairing bug {} ... ".format(file.split(".")[0]))
            print(f"Current bug category: {bug['category']}")
            
            if args.select == "SSelect":
                examples = get_example_from_clusters(args,think_dataset,Select_model,Select_tokenizer,buggy_code)
            elif args.select == "TopK":
                buggy_embedding = vectorizer.transform([buggy_code]).toarray()[0] 
                examples, sim_scores = get_example_topk(args, vectorizer, buggy_code, valid_items, buggy_embedding)
                all_sim_records.append({
                    "file": file,
                    "max_sim": float(max(sim_scores)),
                    "category": bug["category"]
                })
            else:
                examples = list(np.random.choice(list(think_dataset.keys()), args.n_example))
                
            if not examples:
                print(f"Warning: No examples selected for {file}")
                skip_example += 1
                continue
            
            SIMILARITY_THRESHOLD = 0.58 if "humaneval" in args.benchmark_name else (0 if "quixbugs-java" in args.benchmark_name else 0.52) #humaneval0.49 quixbugs=0.59
            if max(sim_scores) < SIMILARITY_THRESHOLD:
                origin_prompt = zero_shot_prompt.format(bug=buggy_code)
                used_zero_shot =  used_zero_shot+1
            else:
                example_bug_1, example_fix_1 = extract_buggy_snippet(
                    think_dataset[examples[0]]["buggy"],
                    think_dataset[examples[0]]["fix"],
                    think_dataset[examples[0]].get("location", []),
                    tokenizer
                )
                example_bug_2, example_fix_2 = extract_buggy_snippet(
                        think_dataset[examples[1]]["buggy"],
                        think_dataset[examples[1]]["fix"],
                        think_dataset[examples[1]].get("location", []),
                        tokenizer
                )     
                origin_prompt = repair_prompt.format(example_bug_1=example_bug_1,example_fix_1=example_fix_1,example_bug_2=example_bug_2,example_fix_2=example_fix_2,bug=buggy_code)
        
        elif args.enhance_methods == "cot":
            origin_prompt = repair_prompt.format(bug=buggy_code)
        print(f"origin_prompt length: {len(origin_prompt)}")
        prompt = prompt_text(origin_prompt, args.model_type, args.enhance_methods)
        print("="*80)
        print("BUG:", file)
        print("prompt chars:", len(origin_prompt))
        print("prompt preview:", origin_prompt[:500])
        print("="*80)
                   
        repair_results = repair.get("data", {}).get(file, {}).get("output", [])
        if len(repair_results) >= args.num_output:
            print(f"Skip {file}")
            continue

        print(f"Starting generation for {file}...")  
        buggy_length = len(tokenizer.encode(buggy_code)) if tokenizer else 200
        max_new_tokens = min(1024, buggy_length + 300) if args.enhance_methods== "cot" else buggy_length + 100
        if 'gpt' in args.model_type:
                for retry in range(3):
                    try:
                        raw_outputs = client.chat.completions.create(
                            model=args.model_type,
                            messages = [
                                {"role": "system", "content": prompt}, 
                                {"role": "user", "content": origin_prompt}
                            ],
                            max_tokens=max_new_tokens,
                            n=args.num_output,
                            temperature= 0.8,                       
                            top_p= 0.95, 
                            timeout=150
                        )
                        break
                    except Exception as e:
                        print(f"{file} retry {retry+1}: {e}")
                        if retry == 2:
                            raise
                        time.sleep(10)
                for choice in raw_outputs.choices:
                    output = choice.message.content
                    if "// End" in output:
                        output = output[:output.find("// End")].strip()
                    if "}" in output:
                        output = output[:output.rfind("}")+1]
                    repair_results.append(output)
        else:
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            buggy_length = tokenizer.encode(buggy_code, return_tensors="pt").to(model.device).shape[1]
            total_input_tokens = inputs.shape[1]
            model_max_length = tokenizer.model_max_length #
            if total_input_tokens + buggy_length + 100 >= model_max_length:
                print(f"Skipping {file}: token length {total_input_tokens} exceeded")
                repair_results.append({"output": "# Token size exceeded.", "valid": False})
                skip_length += 1
                continue
                
            for _ in range(args.num_output - len(repair_results)):
                raw_outputs = model.generate(
                        inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        top_p= 0.95,
                        temperature= 0.8,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id
                    )
                output = tokenizer.decode(raw_outputs[0][len(inputs[0]):])
                if "// End" in output:
                    output = output[:output.find("// End")].strip()
                if "}" in output:
                    output = output[:output.rfind("}") + 1]
                repair_results.append(output)
        
        if "data" not in repair:
                repair["data"] = {}

        repair["data"][file] = {
                "input": buggy_code,
                "function range": bug.get("function range", ""),
                "category": bug["category"],
                "output": repair_results,
                "examples": locals().get("examples", []),
        }

        print(f"Saved {file}, total outputs: {len(repair_results)}")
        with open(output_data_path, 'w') as f:
            json.dump(repair, f, indent=4)
        print(f"Saving repair results to: {output_data_path}")
           
    df = pd.DataFrame(all_sim_records)
    print(df["max_sim"].describe())
    suspicious = df[df["max_sim"] > 0.9]
    print(suspicious)

def load_json_data(path):
    with open(path, "r") as f:
        content = json.load(f)
    return content["data"] if "data" in content else content

def clean_parse_d4j(rag_dataset_dir):  
    result = load_json_data(os.path.join(rag_dataset_dir, "defects4j.json"))
    cleaned_result = {}
    for k, v in result.items():
        lines = v['buggy'].splitlines()
        leading_white_space = len(lines[0]) - len(lines[0].lstrip())
        cleaned_result[k + ".java"] = {"buggy": "\n".join([line[leading_white_space:] for line in lines])}
        lines = v['fix'].splitlines()
        leading_white_space = len(lines[0]) - len(lines[0].lstrip())
        cleaned_result[k + ".java"]["fix"] = "\n".join([line[leading_white_space:] for line in lines])
        cleaned_result[k + ".java"]["location"] = [location - v['start'] + 1 for location in v["location"]]
        cleaned_result[k + ".java"]["category"] = v["category"]
    return cleaned_result

def build_datasets(args):
    print(f"Target dataset: {args.benchmark_name}")
    inference_dataset = load_json_data(args.benchmark_data)
    rag_dataset_dir = os.path.join(PROJECT_ROOT, "datasets", "rag", "Datasets")
    d4j_dataset = clean_parse_d4j(rag_dataset_dir)
    if "defects4j" in args.benchmark_name:
        def match_d4j_key(key):
            return re.sub(r'\.java$', '', key)
        c1_projects = set()
        for key in inference_dataset:
            m = re.match(r"^(\w+)_(\d+)", key)
            if m:
                c1_projects.add(f"{m.group(1)}-{m.group(2)}")
        think_dataset = {key: value for key, value in d4j_dataset.items() if match_d4j_key(key) not in c1_projects}
        
    elif "quixbugs" in args.benchmark_name:
        think_dataset = load_json_data(os.path.join(rag_dataset_dir, "humaneval.json"))
    elif "humaneval-java" in args.benchmark_name:
        think_dataset = load_json_data(os.path.join(rag_dataset_dir, "quixbugs.json"))
    else:
        raise ValueError(f"Unknown dataset: {args.benchmark_name}")

    print(f"Successfully loaded {len(inference_dataset)} cases for inference.")
    print(f"Successfully loaded {len(think_dataset)} cases for think/retrieval pool.")
    
    return inference_dataset, think_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_name", type=str, default="D4JV1.2",help="Dataset to use, current support: D4JV1.2, D4JV2.0, RWBV2.0")
    parser.add_argument("--benchmark_data", type=str, default="defects4j", help="Dataset to use for clustering and example selection, current support: defects4j, quixbugs-java, humaneval")
    parser.add_argument("--select", type=str, default="TopK", help="Selection strategy to use, current support: TopK, SSelect, RSelect")
    parser.add_argument('--model_type', help='model to use for code translation. should be one of [CodeGeeX,StarCoder,CodeGen,TB-Airoboros,TB-Vicuna,LLaMa,CodeLLama]', required=True, type=str)
    parser.add_argument("--n_example", type=int, default=2)
    parser.add_argument("--num_output", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--enhance_methods', type=str, default="rag")
    # model
    parser.add_argument("--model_name_or_path", type=str, help="The model checkpoint for weights initialization.")       
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--output_file_name", default="output.json", type=str)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1)

    args = parser.parse_args()
    set_seed(args.seed)
    
    model, tokenizer = None, None
    if 'gpt' not in args.model_type:
        model, tokenizer = create_model_and_tokenizer(args.model_name_or_path, args.model_type)
        
    output_data_path = os.path.join(args.output_dir, args.output_file_name)
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    
    kb_dir = os.path.join(PROJECT_ROOT,"datasets","rag",args.model_type,args.benchmark_name,)
    os.makedirs(kb_dir, exist_ok=True)
    args.kb_path = os.path.join(kb_dir, "kb.json")
    args.vec_path = os.path.join(kb_dir, "tfidf_vectorizer.pkl")
    
    inference_dataset, think_dataset = build_datasets(args)
    Select_model, Select_tokenizer = None, None
    if args.select in ["SSelect", "RSelect"]:
        unixcoder_path = os.path.join(PROJECT_ROOT, "models", "unixcoder-base")
        Select_tokenizer = AutoTokenizer.from_pretrained(unixcoder_path)
        Select_model = AutoModel.from_pretrained(unixcoder_path, trust_remote_code=True)
        if torch.cuda.is_available(): Select_model = Select_model.to("cuda")
        get_embedding_UniXcoder(args, Select_model,Select_tokenizer, think_dataset)
    elif args.select == "TopK":
        get_embedding_TFIDF(args, think_dataset)
        with open(args.vec_path,"rb") as f:
            vectorizer = pickle.load(f)
        

    # if torch.cuda.is_available() and getattr(args, 'num_gpus', 1) > 1 and 'gpt' not in args.model_type:
    #     bug_items = sorted(inference_dataset.items())
    #     chunk_size = (len(bug_items) + args.num_gpus - 1) // args.num_gpus
    #     gid = getattr(args, 'gpu_id', 0)
    #     my_chunk = dict(bug_items[gid * chunk_size: (gid + 1) * chunk_size])
    #     inference_dataset = my_chunk
    #     if f"_gpu{gid}" not in args.output_file_name:
    #         args.output_file_name = args.output_file_name.replace(".json", f"_gpu{gid}.json")
    #     print(f"GPU {gid}/{args.num_gpus}: {len(my_chunk)} bugs -> {args.output_file_name}")

    rag_repair(args, Select_model, Select_tokenizer, think_dataset, inference_dataset, tokenizer, model, output_data_path, vectorizer)

if __name__=="__main__":
    main()
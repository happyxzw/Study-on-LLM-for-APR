from __future__ import absolute_import, division, print_function
import os
import re
import sys
import time
sys.path.append('/home/chenshiping/peft4apr/src/benchmarks/inference_and_validation_src')
import json
import torch
import random
import argparse
import warnings
import numpy as np
from sklearn.cluster import KMeans
from utils.build_d4j import build_d4j1_2
from utils.parse_d4j import clean_parse_d4j
from apr_utils import create_model_and_tokenizer,prompt_text,client
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_embedding_UniXcoder(args, model, tokenizer, think_dataset):
    save_dir = f"./Results/{args.model_type}/{args.benchmark_name}"
    os.makedirs(save_dir, exist_ok=True)
    kb_path = os.path.join(save_dir, "kb.json")

    # 如果 KB 已存在且有效，直接加载复用，避免多进程竞争写入
    if os.path.exists(kb_path):
        try:
            with open(kb_path, "r") as f:
                existing = json.load(f)
            # 把已有 embedding 合并到 think_dataset
            for file, sample in think_dataset.items():
                if file in existing and "embedding_UniXcoder" in existing[file]:
                    think_dataset[file]["embedding_UniXcoder"] = existing[file]["embedding_UniXcoder"]
            print(f"Loaded existing UniXcoder embeddings from {kb_path}")
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

    # 原子写入：先写临时文件再 rename，避免多进程竞争损坏
    tmp_path = kb_path + f".tmp{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(think_dataset, f, indent=4)
    os.replace(tmp_path, kb_path)
    print(f"Saved think_dataset with UniXcoder embeddings to {kb_path}")
 

def get_embedding_TFIDF(args, think_dataset):
    save_dir = f"./Results/{args.model_type}/{args.benchmark_name}"
    os.makedirs(save_dir, exist_ok=True)
    kb_path = os.path.join(save_dir, "kb.json")
    vec_path = os.path.join(save_dir, "tfidf_vectorizer.pkl")

    # 如果 KB + vectorizer 已存在且有效，直接加载复用
    if os.path.exists(kb_path) and os.path.exists(vec_path):
        try:
            with open(kb_path, "r") as f:
                existing = json.load(f)
            for file, sample in think_dataset.items():
                if file in existing and "embedding_TFIDF" in existing[file]:
                    think_dataset[file]["embedding_TFIDF"] = existing[file]["embedding_TFIDF"]
            print(f"Loaded existing TF-IDF embeddings from {kb_path}")
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

    # 原子写入 vectorizer
    tmp_vec = vec_path + f".tmp{os.getpid()}"
    with open(tmp_vec, "wb") as f_out:
        pickle.dump(vectorizer, f_out)
    os.replace(tmp_vec, vec_path)

    # 原子写入 KB
    tmp_path = kb_path + f".tmp{os.getpid()}"
    with open(tmp_path, "w") as f_out:
        json.dump(think_dataset, f_out, indent=4)
    os.replace(tmp_path, kb_path)
    print(f"Saved think_dataset with TF-IDF embeddings to {kb_path}")
       
def get_clusters(args, category):
    if args.select == "CSelect":
        embedding_key = "embedding"
    else:
        embedding_key = "embedding_UniXcoder"
        
    kb = json.load(open(f"./Results/{args.model_type}/{args.benchmark_name}/kb.json", "r"))
    
    valid_items = [
        {"file": k, **v}
        for k, v in kb.items()
        #if v["category"] == category
    ]
    embeddings = np.asarray([item[embedding_key] for item in valid_items])
    
    if len(embeddings) == 0:
        return None, None
    kmeans = KMeans(n_clusters=min(args.n_example, len(embeddings)),n_init=10,random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.labels_
    return labels, valid_items

def get_example_from_clusters(args, model, tokenizer, buggy, labels, valid_items):

    print("Get buggy func embedding")
    print(f"model device: {next(model.parameters()).device}")
    if args.select == "CSelect":
        embedding_key = "embedding"
    else:
        embedding_key = "embedding_UniXcoder"
    points = [item["file"] for item in valid_items]
    embeddings = [item[embedding_key] for item in valid_items]
    selected_points = []
    print(f"valid_items count: {len(valid_items)}")
    print(f"embeddings count: {len(embeddings)}")
    print(f"labels unique: {np.unique(labels)}")

    if args.select == "CSelect":
        code_tokens = tokenizer.tokenize(buggy)[:args.block_size-4]
        source_tokens = [
            tokenizer.cls_token,
            "<encoder_only>",
            tokenizer.sep_token
        ] + code_tokens + [tokenizer.sep_token]

        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length

        buggy_embedding = model.get_xcode_vec(
            torch.tensor(source_ids).unsqueeze(0).to(args.device)
        )[0].cpu().detach().numpy()

    else:
        tokenized_code = tokenizer.encode_plus(
            buggy,
            max_length=400,
            truncation=True,
            return_tensors="pt"
        )

        tokenized_code = {k: v.to(model.device)for k, v in tokenized_code.items()}
        outputs = model(**tokenized_code)
        buggy_embedding = outputs[0][0, 0, :].detach().cpu().numpy()

    for i in range(min(args.n_example, len(np.unique(labels)))):

        cluster_points = np.where(labels == i)[0]

        if len(cluster_points) == 0:
            continue

        cluster_embeddings = [
            embeddings[p] for p in cluster_points
        ]

        similarities = cosine_similarity(
            [buggy_embedding],
            cluster_embeddings
        )

        most_similar_index = np.argmax(similarities)

        selected_point = cluster_points[most_similar_index]

        selected_points.append(points[selected_point])
    print(f"selected_points: {selected_points}")
    return selected_points

def get_example_topk(args, model, tokenizer, buggy, valid_items):

    points = [item["file"] for item in valid_items]
    embeddings = [item["embedding_UniXcoder"] for item in valid_items]

    tokenized_code = tokenizer.encode_plus(
        buggy,
        max_length=400,
        truncation=True,
        return_tensors="pt"
    )

    tokenized_code = {
        k: v.to(model.device)
        for k, v in tokenized_code.items()
    }

    outputs = model(**tokenized_code)

    buggy_embedding = outputs[0][0,0,:].detach().cpu().numpy()

    similarities = cosine_similarity(
        [buggy_embedding],
        embeddings
    )[0]

    topk = np.argsort(similarities)[::-1][:args.n_example]

    selected_points = [
        points[i]
        for i in topk
    ]

    return selected_points, similarities[topk]

def get_example_topk_tfidf(args, vectorizer, buggy, valid_items):
    points = [item["file"] for item in valid_items]
    embeddings = np.array([item["embedding_TFIDF"] for item in valid_items])

    buggy_vec = vectorizer.transform([buggy]).toarray()[0]  # 用同一个vectorizer做transform，不是fit

    similarities = cosine_similarity([buggy_vec], embeddings)[0]
    topk = np.argsort(similarities)[::-1][:args.n_example]

    selected_points = [points[i] for i in topk]
    return selected_points, similarities[topk]

def tokenize_code(text):
    if text is None:
        return []
    return text.strip().split()

def add_bug_comments(code_string, buggy_line_numbers):
    lines = code_string.split("\n")
    for line_number in buggy_line_numbers:
        if 1 <= line_number <= len(lines):
            lines[line_number-1] += " // Buggy Line"
    modified_code_string = "\n".join(lines)
    return modified_code_string


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

def rag_repair(args, Select_model, Select_tokenizer, think_dataset, inference_files, tokenizer, model, bm25_engine=None):
    if args.enhance_methods == "rag":
        with open(f"/home/chenshiping/peft4apr/datasets/prompt/repair_prompt_open_source_rag.txt") as f:
            repair_prompt = f.read()
        with open(f"/home/chenshiping/peft4apr/datasets/prompt/repair_prompt_open_source.txt") as f:
            zero_shot_prompt = f.read()
    elif args.enhance_methods == "cot":
        with open(f"/home/chenshiping/peft4apr/datasets/prompt/repair_prompt_cot.txt") as f:
            repair_prompt = f.read()

    with open(f"./Results/{args.model_type}/{args.benchmark_name}/kb.json", 'r') as f:
        kb = json.load(f)
    vectorizer = None
    with open(f"./Results/{args.model_type}/{args.benchmark_name}/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    base_name = re.sub(r'_c[12]$', '', args.benchmark_name)

    try:
        with open(
            f"/home/chenshiping/peft4apr/datasets/results/"
            f"{base_name}/{args.model_type}/{args.enhance_methods}/{args.output_file_name}",
            "r"
        ) as f:
            repair = json.load(f)
    except FileNotFoundError:
        repair = {}
    
    first_bug = list(inference_files.items())[0]
    print(f"First bug in inference_files: {first_bug[0]}")
    print(f"Keys: {first_bug[1].keys()}")  
    total = 0
    skip_no_buggy = 0
    skip_cluster = 0
    skip_example = 0
    skip_length = 0
    used_zero_shot = 0
    all_sim_records = []
    
    for file, bug in inference_files.items():
        total += 1
        buggy_code = bug.get("buggy") or bug.get("input", "")
        if not buggy_code:
            print(f"Warnisssng: No buggy code found for {file}")
            skip_no_buggy += 1
            continue

        if args.enhance_methods == "rag":
            current_bug_type = bug['category']
            labels, valid_items = get_clusters(args, current_bug_type)
            valid_items = [{"file": k, **v} for k, v in kb.items()]  
            if  valid_items is None:
                print(f"Warning: No valid items found for category {current_bug_type} in {file}")
                skip_cluster += 1
                continue
            print("Repairing bug {} ... ".format(file.split(".")[0]))
            print(f"Current bug category: {current_bug_type}")
            if args.select == "SSelect":
                examples = get_example_from_clusters(args,Select_model,Select_tokenizer,buggy_code,labels,valid_items)
                if not examples:
                    print(f"Warning: No examples selected for {file}")
                    skip_example += 1
                    continue
            elif args.select == "TopK":
                examples, sim_scores = get_example_topk(args,Select_model, Select_tokenizer, buggy_code, valid_items)
                all_sim_records.append({
                    "file": file,
                    "max_sim": float(max(sim_scores)),
                    "category": bug["category"]
                })

                
            else:
                examples = np.random.choice(list(kb.keys()), args.n_example)

            print(f"examples: {examples}, sim_scores: {sim_scores}")
            print(f"think_dataset keys sample: {list(think_dataset.keys())[:5]}")
            print(f"examples[0] in think_dataset: {examples[0] in think_dataset}")
            

            SIMILARITY_THRESHOLD = 0.58 if "humaneval" in args.benchmark_name else (0 if "quixbugs-java" in args.benchmark_name else 0.52) #humaneval0.49 quixbugs=0.59
            if max(sim_scores) < SIMILARITY_THRESHOLD:
                origin_prompt = zero_shot_prompt.format(bug=buggy_code)
                used_zero_shot =  used_zero_shot+  1
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
        
        if "data" in repair and file in repair["data"]:
            repair_results = repair["data"][file]["output"]
        else:
            repair_results = []
        if ("data" in repair and file in repair["data"] and len(repair["data"][file]["output"]) >= args.num_output):
            print(f"Skip {file}")
            continue    
        prompt = prompt_text(origin_prompt, args.model_type, args.enhance_methods)
        print("="*80)
        print("BUG:", file)
        print("prompt chars:", len(origin_prompt))
        print("prompt preview:", origin_prompt[:500])
        print("="*80)

        
        repair_result = []
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
                            temperature=args.temperature,                       
                            top_p=args.p, 
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
                        repair_result.append({"output": "# Token size exceeded.", "valid": False})
                        skip_length += 1
                        continue
                
                for _ in range(args.num_output - len(repair_results)):
                    raw_outputs = model.generate(
                        inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        top_p=args.p,
                        temperature=args.temperature,
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

        print(f"Saving repair results to: {args.output_dir}{args.output_file_name}")
        with open(os.path.join(args.output_dir, args.output_file_name), 'w') as f:
            json.dump(repair, f, indent=4)
        print(f"Saved {file}, total outputs: {len(repair_results)}")
        
   
    import pandas as pd
    df = pd.DataFrame(all_sim_records)
    print(df["max_sim"].describe())
    suspicious = df[df["max_sim"] > 0.9]
    print(suspicious)
    df.to_csv("tfidf_pipeline_sim_distribution.csv", index=False)
    print(f"总计: {total}")
    print(f"零样本推理使用: {used_zero_shot}")
    print(f"无buggy代码跳过: {skip_no_buggy}")
    print(f"无同类别样本跳过: {skip_cluster}")
    print(f"无示例跳过: {skip_example}")
    print(f"token超长跳过: {skip_length}")
    print(f"成功处理: {total - skip_no_buggy - skip_cluster - skip_example - skip_length}")


def load_json_data(path):
    with open(path, "r") as f:
        content = json.load(f)
    return content["data"] if "data" in content else content

def build_datasets(args):
    d4j_dataset = clean_parse_d4j(folder="./Datasets/")
    print(f"Target dataset: {args.benchmark_name}")
    inference_dataset = load_json_data(args.benchmark_data)
    if "defects4j" in args.benchmark_name:
        pool1 = load_json_data("/home/chenshiping/peft4apr/src/ThinkRepair-main/Datasets/quixbugs.json")
        pool2 = load_json_data("/home/chenshiping/peft4apr/src/ThinkRepair-main/Datasets/humaneval.json")
        def match_d4j_key(key):
            return re.sub(r'\.java$', '', key)
        c1_projects = set()
        c1_data = json.load(open("/home/chenshiping/peft4apr/datasets/benchmarks/defects4j_c1.json"))
        for k in c1_data.get("data", {}).keys():
            m = re.match(r'^(\w+)_(\d+)', k)
            if m:
                c1_projects.add(f"{m.group(1)}-{m.group(2)}")
        pool3 = {key: value for key, value in d4j_dataset.items() if match_d4j_key(key) not in c1_projects}
        print(f"pool3 matched: {len(pool3)} projects")
        print(list(pool3.keys())[:10])
        # think_dataset = {**pool1, **pool2, **pool3}
        think_dataset = pool3

    elif "quixbugs-java" in args.benchmark_name:
        pool1 = d4j_dataset
        pool2 = load_json_data("/home/chenshiping/peft4apr/src/ThinkRepair-main/Datasets/humaneval.json")
        think_dataset = pool2

    elif "humaneval" in args.benchmark_name:
        pool1 = load_json_data("/home/chenshiping/peft4apr/condefects_java_rag.json")
        pool2 = load_json_data("/home/chenshiping/peft4apr/src/ThinkRepair-main/Datasets/quixbugs.json")
        think_dataset = {**pool2}

    else:
        raise ValueError(f"Unknown dataset: {args.benchmark_name}")

    print(f"Successfully loaded {len(inference_dataset)} cases for inference.")
    print(f"Successfully loaded {len(think_dataset)} cases for think/retrieval pool.")
    
    return inference_dataset, think_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_name", type=str, default="D4JV1.2",help="Dataset to use, current support: D4JV1.2, D4JV2.0, RWBV2.0")
    parser.add_argument("--benchmark_data", type=str, default="defects4j", help="Dataset to use for clustering and example selection, current support: defects4j, quixbugs-java, humaneval")
    parser.add_argument("--select", type=str, default="CSelect",
                        help="Selection strategy to use, current support: CSelect, SSelect, RSelect")
    parser.add_argument('--model_type', help='model to use for code translation. should be one of [CodeGeeX,StarCoder,CodeGen,TB-Airoboros,TB-Vicuna,LLaMa,CodeLLama]', required=True, type=str)
    parser.add_argument("--n_example", type=int, default=2)
    parser.add_argument("--num_output", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--enhance_methods', type=str, default="rag")
    parser.add_argument('--p', help='Only the most probable tokens with probabilities that add up to top_p or higher are considered during decoding. The valid range is 0.0 to 1.0. 1.0 is equivalent to disabled and is the default. Only applies to sampling mode. Also known as nucleus sampling.', type=float, default=0.95)
    parser.add_argument('--temperature', help='A value used to warp next-token probabilities in sampling mode. Values less than 1.0 sharpen the probability distribution, resulting in "less random" output. Values greater than 1.0 flatten the probability distribution, resulting in "more random" output. A value of 1.0 has no effect and is the default. The allowed range is 0.0 to 2.0.', type=float, default=0.8)
    
    # model
    parser.add_argument("--model_dir", default="saved_models", type=str)
    parser.add_argument("--model_name_or_path", type=str, help="The model checkpoint for weights initialization.")
    parser.add_argument("--block_size", default=400, type=int,
                        help="Optional input sequence length after tokenization.")         
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--output_file_name", default="output.json", type=str)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1)

    args = parser.parse_args()

    print(f"CUDA available: {torch.cuda.is_available()}, device_count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    set_seed(args.seed)
    if 'gpt' in args.model_type:
        model, tokenizer = None, None
    else:
        model, tokenizer = create_model_and_tokenizer(args.model_name_or_path, args.model_type)
    os.makedirs(args.output_dir, exist_ok=True)
    inference_dataset, think_dataset = build_datasets(args)
    kb_path = f"./Results/{args.model_type}/{args.benchmark_name}/kb.json"
    if os.path.exists(kb_path):
        with open(kb_path, 'r') as f:
            existing_kb = json.load(f)
        for file in think_dataset:
            if file in existing_kb and "embedding_UniXcoder" in existing_kb[file]:
                think_dataset[file]["embedding_UniXcoder"] = existing_kb[file]["embedding_UniXcoder"]
    
    bm25_engine = None
    Select_model, Select_tokenizer = None, None
    if args.select in ["CSelect", "SSelect", "RSelect", "TopK"]:
        Select_tokenizer = AutoTokenizer.from_pretrained("/home/chenshiping/models/unixcoder-base")
        Select_model = AutoModel.from_pretrained("/home/chenshiping/models/unixcoder-base", trust_remote_code=True)
        if torch.cuda.is_available(): Select_model = Select_model.to("cuda")
        get_embedding_UniXcoder(args, Select_model,Select_tokenizer, think_dataset)
        

    if torch.cuda.is_available() and getattr(args, 'num_gpus', 1) > 1 and 'gpt' not in args.model_type:
        bug_items = sorted(inference_dataset.items())
        chunk_size = (len(bug_items) + args.num_gpus - 1) // args.num_gpus
        gid = getattr(args, 'gpu_id', 0)
        my_chunk = dict(bug_items[gid * chunk_size: (gid + 1) * chunk_size])
        inference_dataset = my_chunk
        if f"_gpu{gid}" not in args.output_file_name:
            args.output_file_name = args.output_file_name.replace(".json", f"_gpu{gid}.json")
        print(f"GPU {gid}/{args.num_gpus}: {len(my_chunk)} bugs -> {args.output_file_name}")

    rag_repair(args, Select_model, Select_tokenizer, think_dataset, inference_dataset, tokenizer, model, bm25_engine=bm25_engine)


if __name__=="__main__":
    main()
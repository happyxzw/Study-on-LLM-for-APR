import fire
import datetime
from humaneval_patch_validate import validate_humaneval
from quixbugs_patch_validate import validate_quixbugs
from  defects4j_patch_validate import validate_defects4j
from result_look import cal_result
import re
import json


def get_java_method_name(line: str):
    line = line.strip()

    pattern = re.compile(
        r'^(?:public|private|protected)?\s*'
        r'(?:static\s+)?'
        r'(?:final\s+)?'
        r'(?:synchronized\s+)?'
        r'(?:<[^>]+>\s+)?'                 # 方法泛型，如 <T>
        r'[\w<>\[\], ?]+\s+'               # 返回类型，如 Integer[] / List<String>
        r'([A-Za-z_]\w*)\s*\('             # 方法名
    )

    m = pattern.match(line)
    return m.group(1) if m else None


def extract_function(text, func_signature=None):
    if not text:
        return ""

    # 去掉 tag
    text = re.sub(r"\[/?[A-Z][A-Z0-9_ ]*\]", "", text)

    # 截断到第一个 [INST] 或 ### 之前
    for stop in ['[INST]', '[/INST]', '###']:
        if stop in text:
            text = text.split(stop)[0]

    lines = text.split("\n")
    start = None

    # 如果指定了目标函数，优先匹配目标函数
    if func_signature:
        found_target = False
        for i, line in enumerate(lines):
            l = line.strip()
            method_name = get_java_method_name(l)
            if method_name == func_signature:
                start = i
                found_target = True
                break
        if not found_target:
            return ""
    else:
        # 没指定时取第一个方法
        for i, line in enumerate(lines):
            l = line.strip()
            method_name = get_java_method_name(l)
            if method_name:
                start = i
                break

    if start is None:
        return ""

    brace = 0
    started = False
    result = []

    for line in lines[start:]:
        result.append(line)
        brace += line.count("{")
        brace -= line.count("}")
        if "{" in line:
            started = True
        if started and brace == 0:
            break

    func = "\n".join(result).strip()

    # 截到最后一个 }
    last_brace = func.rfind("}")
    if last_brace != -1:
        func = func[:last_brace + 1]

    # 后置校验
    if func_signature:
        first_line = func.split("\n")[0].strip()
        extracted_name = get_java_method_name(first_line)
        if extracted_name != func_signature:
            return ""

    return func


def is_valid_function(code):
    if not code:
        return False
    if len(code) < 20:
        return False
    if "return" not in code and ";" not in code:
        return False

    bad_keywords = [
        "The following function",
        "Example",
        "###",
        "instruction",
    ]
    for k in bad_keywords:
        if k.lower() in code.lower():
            return False

    return True


def clean_output(output, func_signature=None):
    code = extract_function(output, func_signature=func_signature)
    if is_valid_function(code):
        return code
    return ""


def preprocess_outputs(input_file):
    data = json.load(open(input_file, 'r', encoding='utf-8'))

    for k in data["data"]:
        input_text = data["data"][k].get("input", "")
        func_signature = None

        # 从 input 中提取函数名
        for line in input_text.split("\n"):
            method_name = get_java_method_name(line)
            if method_name:
                func_signature = method_name
                break

        outputs = data["data"][k]["output"]
        cleaned_outputs = []

        for o in outputs:
            c = extract_function(o, func_signature=func_signature)
            if is_valid_function(c) and "[INST]" not in c:
                cleaned_outputs.append(c)

        data["data"][k]["output"] = cleaned_outputs

    tmp_file = input_file.replace(".json", "_cleaned.json")
    json.dump(data, open(tmp_file, "w", encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"Preprocessed outputs saved to {tmp_file}")
    return tmp_file

def peft_patch_validation(
    input_file: str = " llmpeft4apr/results/deepseek_coder_6.7b_base_apr_p-tuning_on_humaneval_output_24_03_31_13_27_27.json", 
    output_dir: str = " llmpeft4apr/results/", #validation results
    # copy from  llmpeft4apr/validation_benchmark_dataset/benchmarks/humaneval-java
    benchmark_dir: str = " llmpeft4apr/validation_benchmark_dataset/benchmarks/humaneval-java/", #test suits
    benchmark_name: str = "humaneval",
    model_type: str = "deepseek-coder-6.7b-base",
    enhance_methods: str = "lora",
    train_dataset: str = "apr",
    validation_time: str = "",
    validation_file: str = "",
    log_file: str = "validation_logs"    
):
    input_file = preprocess_outputs(input_file)
    validate_fp = ''
    if 'humaneval' in benchmark_name:
        validate_fp = validate_humaneval(
                input_file= input_file,
                output_dir=output_dir,
                benchmark_dir=benchmark_dir,
                benchmark_name=benchmark_name,
                model_type=model_type,
                peft_type=enhance_methods,
                train_dataset=train_dataset,
                validation_file=validation_file,
                log_file=log_file
            )
        
        # cal_result(validate_fp)
    elif "quixbugs-java" in benchmark_name:
        print(benchmark_dir)
        validate_fp =validate_quixbugs(
            input_file= input_file,
            output_dir=output_dir,
            benchmark_name=benchmark_name,
            model_type=model_type,
            peft_type=enhance_methods,
            tmp_dir=benchmark_dir,
            train_dataset=train_dataset,
            validation_file=validation_file,
            log_file=log_file
        )
    elif benchmark_name == 'defects4j':
        print(validation_time[1:])
        validate_fp =validate_defects4j(
            input_file= input_file,
            output_dir=output_dir,
            benchmark_dir=benchmark_dir,
            benchmark_name=benchmark_name,
            model_type=model_type,
            peft_type=enhance_methods,
            tmp_dir=benchmark_dir,
            train_dataset=train_dataset,
            validation_file=validation_file,
            log_file=log_file
        )
    else:
        print('Wrong benchmark name!')
    print(f"log files saved to {log_file}")
    if validate_fp != '':
        cal_result(validate_fp, model_type, enhance_methods, train_dataset, benchmark_name)
if __name__ == '__main__':
    fire.Fire(peft_patch_validation)
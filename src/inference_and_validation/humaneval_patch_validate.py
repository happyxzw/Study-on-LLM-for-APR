import json
import os
import re
import shutil
import time
import subprocess
import fire
from datetime import datetime
from pathlib import Path
from validation_utils import write_log, exec_command_with_timeout, parse_test_output,  save_json, prepare_validation_environment, is_validated

def clean_and_create_benchmark_dir(benchmark_dir, benchmark_name):
    buggy_dir = benchmark_dir / "src" / "main" / "java" / benchmark_name / "buggy"
    test_dir = benchmark_dir / "src" / "test" / "java" / benchmark_name
    
    shutil.rmtree(buggy_dir, ignore_errors=True)
    shutil.rmtree(test_dir, ignore_errors=True)
    buggy_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

def parse_function_range(range_str):
    start, end = range_str.split('-')
    start = int(start.split(',')[0])
    end = int(end.split(',')[0])
    return start, end

def validate_humaneval(
    input_file: str = "llmpeft4apr/results/deepseek_coder_6.7b_base_apr_p-tuning_on_humaneval_output.json",
    output_dir: str = "llmpeft4apr/results/",
    benchmark_dir: str = "llmpeft4apr/validation_benchmark_dataset/benchmarks/humaneval-java/",
    benchmark_name: str = "humaneval",
    model_type: str = "deepseek-coder-6.7b-base",
    peft_type: str = "lora",
    train_dataset: str = "apr",
    validation_file: str = "",
    log_file: str = "validation_logs"
):

    benchmark_dir = Path(benchmark_dir)
    output_dir = Path(output_dir)
    log_file = Path(log_file)
    clean_and_create_benchmark_dir(benchmark_dir, benchmark_name)  
    validation_file, model_output,validated_result, plausible, total = prepare_validation_environment(
        input_file=input_file,output_dir=output_dir,benchmark_name=benchmark_name,model_type=model_type,peft_type=peft_type,train_dataset=train_dataset,validation_file=validation_file,log_file=log_file)
    
    for proj in model_output['data']:
        print('start validating', proj)
        project_log_file = os.path.join(log_file,f"{proj}.log")
         
        if 'output' not in model_output['data'][proj]:
            continue
        else:
            expected_count = len(model_output['data'][proj]['output'])
        if proj in validated_result['data'] and is_validated(validated_result['data'][proj], expected_count):
            print('validation result existed!')
            continue
        validated_result['data'][proj] = {}
        validated_result['data'][proj]['output'] = []
        src_buggy_dir = benchmark_dir / "src" / "main" / "java" / benchmark_name / "buggy"
        src_test_dir = benchmark_dir / "src" / "test" / "java" / benchmark_name
        os.makedirs(src_buggy_dir, exist_ok=True)
        os.makedirs(src_test_dir, exist_ok=True)

        # 把原始缺陷代码和测试代码从备份区搬过来
        shutil.copyfile(benchmark_dir / "src_bak" / "main" / "java" / benchmark_name / "buggy" / f"{proj}.java", src_buggy_dir / f"{proj}.java")
        shutil.copyfile(benchmark_dir / "src_bak" / "test" / "java" / benchmark_name / f"TEST_{proj}.java", src_test_dir / f"TEST_{proj}.java")
        baseline_result = humaneval_test_suite(proj,benchmark_dir)
        baseline_info = parse_test_output(baseline_result["stdout"],baseline_result["stderr"])
        baseline_failed_tests = set(baseline_info["failed_testcases"])
        print("Baseline failed tests:", baseline_failed_tests)
        for key, value in model_output['data'][proj].items():
            if key != 'output':
                validated_result['data'][proj][key] = value

        validated_result['data'][proj]['output'] = []

        total += 1
        current_is_correct = False

        shutil.copyfile(benchmark_dir / "src_bak" / "test" / "java" / benchmark_name / f"TEST_{proj}.java", benchmark_dir / "src" / "test" / "java" / benchmark_name / f"TEST_{proj}.java")

        for rank, item in enumerate(model_output['data'][proj]['output']):
            shutil.rmtree(benchmark_dir / "target", ignore_errors=True)
            
            start_time = time.time()
            patch = item['patch'] if isinstance(item, dict) else item

            if "deepseek-coder" in model_type:
                end_bucket = patch.rfind('}')
                patch = patch[:end_bucket+1]

            path=benchmark_dir  / "src_bak" / "main" / "java" / benchmark_name / "buggy" / f"{proj}.java"
            
            with open(path, 'r') as f:
                buggy_code = f.read()

            buggy_lines = buggy_code.split('\n')

            start_line_index, end_line_index = parse_function_range(model_output['data'][proj]['function range'])

            patch_prefix = '\n'.join(buggy_lines[:start_line_index-1])
            patch_suffix = '\n'.join(buggy_lines[end_line_index:])

            filename = benchmark_dir / "src" / "main" / "java" / benchmark_name / "buggy" / f"{proj}.java"

            with open(filename, 'w') as f:
                f.write(patch_prefix + '\n' + patch + '\n' + patch_suffix)

            test_result = humaneval_test_suite(proj, benchmark_dir)
            correctness = test_result["correctness"]
            test_stdout = test_result["stdout"]
            test_stderr = test_result["stderr"]
            failure_reason = test_result["failure_reason"]
            
            if correctness == 'plausible' and not current_is_correct:
                plausible += 1
                current_is_correct = True
            print(plausible, total, rank,f"{correctness.capitalize()} patch:",patch)
                
            test_info = parse_test_output(test_stdout,test_stderr)
            current_failed_tests = set(test_info["failed_testcases"])
            print("Current failed tests:", current_failed_tests)
            new_failed_tests = list(current_failed_tests - baseline_failed_tests)
            fixed_failed_tests = list(baseline_failed_tests - current_failed_tests)
            unchanged_failed_tests = list(current_failed_tests & baseline_failed_tests)
            
            if len(new_failed_tests) > 0 and correctness == 'wrong':
                failure_reason = 'Regression'
            elif len(fixed_failed_tests) > 0 and correctness == 'wrong':
                failure_reason = 'Partially fixed'
            elif current_failed_tests == baseline_failed_tests and correctness == 'wrong': 
                failure_reason = 'Unchanged failed tests'
            elapsed_time = time.time() - start_time
            write_log(project_log_file, proj, rank, correctness, elapsed_time, patch,test_stdout, test_stderr, failure_reason=failure_reason, new_failed_tests=new_failed_tests, exceptions=test_info['exceptions'])
            validated_result['data'][proj]['output'].append({
                'rank': rank,
                'patch': patch,
                'correctness': correctness,
                'validation_time': elapsed_time,
                'failure_reason': failure_reason,      
            })

        shutil.rmtree(benchmark_dir / "src" / "main" / "java" / benchmark_name / "buggy/", ignore_errors=True)
        shutil.rmtree(benchmark_dir / "src" / "test" / "java" / benchmark_name, ignore_errors=True)
        os.makedirs(benchmark_dir / "src" / "main" / "java" / benchmark_name / "buggy/", exist_ok=True)
        os.makedirs(benchmark_dir / "src" / "test" / "java" / benchmark_name , exist_ok=True)

        save_json(validated_result, validation_file)
    return validation_file

def humaneval_test_suite(algo, benchmark_dir):
    result = {"correctness": "uncompilable", "failure_reason":None, "stdout": "", "stderr": ""}
    try:
        out, err = exec_command_with_timeout(["mvn", "test", "-Dtest=TEST_" + algo.upper(), "-o"],timeout=20, cwd=benchmark_dir)
        result["stdout"] = out
        result["stderr"] = err
        msg = (str(out)+"\n" + str(err)).lower()
        if "compilation problems" in msg or "compilation failure" in msg:
            result["correctness"] = 'uncompilable'
            result["failure_reason"] = 'Compilation Error'
        elif "timeout" in msg:
            result["correctness"] = 'timeout'
            result["failure_reason"] = 'Timeout'
        elif "build success" in msg and "tests run:" in msg:
            result["correctness"] = 'plausible'
            result["failure_reason"] = None
        else:
            result["correctness"] = 'wrong'
    except Exception as e:
        print(e)
        result["stderr"] = str(e)
    return result
if __name__ == '__main__':
    fire.Fire(validate_humaneval)

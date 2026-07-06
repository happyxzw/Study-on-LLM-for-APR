import json
import re
import os
import shutil
import time
import subprocess
import fire
from datetime import datetime
from pathlib import Path
from validation_utils import write_log, exec_command_with_timeout,is_validated, parse_test_output, save_json, prepare_validation_environment

LIBS_DIR = "/home/chenshiping/peft4apr/jasper/lib"
    
def find_last_closed_brace_index(s):
    stack = []
    end_flag = False
    last_closed_brace_index = -1
    for i, char in enumerate(s):
        if char == '{':
            end_flag = True
            stack.append(i)
        elif char == '}':
            if len(stack) == 0:
                return -1
            stack.pop()
            last_closed_brace_index = i
        if len(stack) == 0 and end_flag:
            return last_closed_brace_index
    return -1

def quixbugs_test_suite(algo, quixbugs_dir):
    JAR_DIR = 'libs/'
    CLASSPATH = f".:java_programs:{JAR_DIR}junit-4.12.jar:{JAR_DIR}hamcrest-core-1.3.jar"
    result = {"correctness": None, "failure_reason":None, "stdout": "", "stderr": ""}
    try:
        test_file = f"java_testcases/junit/{algo.upper()}_TEST.java"
        compile_proc = subprocess.run(["javac", "-cp", CLASSPATH, test_file],cwd=quixbugs_dir,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
        if compile_proc.returncode != 0:
            result["correctness"] = 'uncompilable'
            result["stdout"] = compile_proc.stdout
            result["stderr"] = compile_proc.stderr
            result["failure_reason"] = 'Compilation Error'
            return result
        
        out, err = exec_command_with_timeout(["java", "-cp", CLASSPATH,"org.junit.runner.JUnitCore", f"java_testcases.junit.{algo.upper()}_TEST"], timeout=5, cwd=quixbugs_dir)
        result["stdout"] = out
        result["stderr"] = err
        output=out + err
        
        if "FAILURES" in output:
            result["correctness"] = 'wrong'
        elif "TIMEOUT" in output:
            result["correctness"] = 'timeout'
            result["failure_reason"] = 'Test timeout'
        elif "OK (" in output:
            result["correctness"] = "plausible"
            
    except Exception as e:
        print(e)
        result["stderr"] = str(e)

    return result
    
def copy_java_files(src_dir, dst_dir, proj):
    dst_dir.mkdir(exist_ok=True)
    files = [f"{proj}.java", "Node.java", "WeightedEdge.java"]
    for f in files:
        shutil.copyfile(src_dir / f, dst_dir / f)

def validate_quixbugs(
    input_file: str = " llmpeft4apr/codellama_7b_hf/result/CodeLlama_7b_hf_lora_on_quixbugs-java_output_patches_1709040073.5193715.json", 
    output_dir: str = " llmpeft4apr/codellama_7b_hf/result/", #validation results
    benchmark_name: str = "quixbugs-java",
    model_type: str = "CodeLlama-7b-hf",
    peft_type: str = "lora",
    tmp_dir: str = ' llmpeft4apr/validation_benchmark_dataset/benchmarks/quixbugs_tmp',
    train_dataset: str = "apr",
    validation_file: str = "",
    log_file: str = "validation_logs"
    ):
    
    tmp_path = Path(tmp_dir)
    tmp_path.mkdir(parents=True, exist_ok=True)
    java_programs = tmp_path / "java_programs"
    java_programs_bak = tmp_path / "java_programs_bak"
    
    shutil.copytree(LIBS_DIR, tmp_path / "libs", dirs_exist_ok=True)
    # bug_loc_map = create_bug_loc_map(bug_locs_dir, benchmark_name)
    # if not os.path.exists(tmp_dir):
    #     exec_command_with_timeout(['mkdir', tmp_dir])

    validation_file, model_output,validated_result, plausible, total = prepare_validation_environment(
        input_file=input_file,output_dir=output_dir,benchmark_name=benchmark_name,model_type=model_type,peft_type=peft_type,train_dataset=train_dataset,validation_file=validation_file,log_file=log_file)

    for proj in model_output['data']:
        print('start validating', proj)
        buggy_file = java_programs_bak / f"{proj}.java"   
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

        with open(buggy_file, 'r') as f:
            bak_buggy_code = f.read()
            
        exec_command_with_timeout(['rm', '-rf', str(java_programs)])
        exec_command_with_timeout(['mkdir', str(java_programs)])

        copy_java_files(java_programs_bak, java_programs, proj)
        baseline_result = quixbugs_test_suite(proj,quixbugs_dir=str(tmp_dir))
        baseline_info = parse_test_output(baseline_result["stdout"],baseline_result["stderr"])
        baseline_failed_tests = set(baseline_info["failed_testcases"])

        for key, value in model_output['data'][proj].items():
            if key != 'output':
                validated_result['data'][proj][key] = value

        validated_result['data'][proj]['output'] = []
        total += 1
        current_is_correct = False

        for rank, item in enumerate(model_output['data'][proj]['output']):
            start_time = time.time()
            exec_command_with_timeout(["find", str(java_programs), "-name", "*.class", "-delete"])
            patch = item['patch'] if isinstance(item, dict) else item
            if model_output['data'][proj]['input'] == "":
                continue

            buggy_code_lines = model_output['data'][proj]['input'].strip().split('\n')
            
            patch_prefix_start_index = bak_buggy_code.find(buggy_code_lines[0])
            patch_prefix_end_index = patch_prefix_start_index + find_last_closed_brace_index(
                bak_buggy_code[patch_prefix_start_index:]
            )

            filename = java_programs / f"{proj}.java"
            with open(filename, 'w') as f:
                f.write(
                    bak_buggy_code[:patch_prefix_start_index] +
                    patch +
                    bak_buggy_code[patch_prefix_end_index+1:]
                )
                
            test_result = quixbugs_test_suite(proj,quixbugs_dir=str(tmp_dir))
            correctness = test_result["correctness"]
            test_stdout = test_result["stdout"]
            test_stderr = test_result["stderr"]
            failure_reason = test_result["failure_reason"]
            print("STDERR:", test_stderr)
            if correctness == 'plausible' and not current_is_correct:
                plausible += 1
                current_is_correct = True
            print(plausible, total, rank,f"{correctness.capitalize()} patch:",patch)
                
            test_info = parse_test_output(test_stdout,test_stderr)
            current_failed_tests = set(test_info["failed_testcases"])
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
            write_log(project_log_file, proj, rank, correctness, elapsed_time, patch, test_stdout, test_stderr, failure_reason=failure_reason, new_failed_tests=new_failed_tests, exceptions=test_info['exceptions'])
            validated_result['data'][proj]['output'].append({
                'rank': rank,
                'patch': patch,
                'correctness': correctness,
                'validation_time': elapsed_time,
                'failure_reason': failure_reason,      
            })
            
            exec_command_with_timeout(["find", str(java_programs), "-name", "*.class", "-delete"])
            
    save_json(validated_result, validation_file)
    return validation_file

if __name__ == '__main__':
    fire.Fire(validate_quixbugs)
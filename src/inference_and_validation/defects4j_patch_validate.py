import json
import sys
import re
import os
import shutil
import time
import subprocess
import fire
from uuid import uuid4
from datetime import datetime
from validation_utils import write_log, exec_command_with_timeout, save_json, prepare_validation_environment, is_validated, parse_test_output

def defects4j_test_suite(project_dir, timeout=300):
    out, err = exec_command_with_timeout(["defects4j", "test", "-r"], timeout, cwd=project_dir)
    return out, err

def checkout_defects4j_project(project, bug_id, tmp_dir):
    subprocess.run(["defects4j", "checkout", "-p", project, "-v", bug_id, "-w", tmp_dir],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

    
def clean_tmp_folder(tmp_dir):
    if os.path.isdir(tmp_dir):
        for files in os.listdir(tmp_dir):
            file_p = os.path.join(tmp_dir, files)
            try:
                if os.path.isfile(file_p):
                    os.unlink(file_p)
                elif os.path.isdir(file_p):
                    shutil.rmtree(file_p)
            except Exception as e:
                raise e
    else:
        os.makedirs(tmp_dir)
        
def compile_fix(project_dir):
    p = subprocess.Popen(["defects4j", "compile"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=project_dir)
    out, err = p.communicate()
    if "FAIL" in str(err) or "FAIL" in str(out):
        return False
    return True

def defects4j_trigger(project_dir, timeout=300):
    out, err = exec_command_with_timeout(["defects4j", "export", "-p", "tests.trigger"], timeout, cwd=project_dir)
    return out, err

def validate_defects4j(
    input_file: str = "",
    output_dir: str = "",
    benchmark_dir: str = "",
    benchmark_name: str = "defects4j",
    model_type: str = "CodeLlama-7b-hf",
    enhancement_type: str = "zero-shot",
    tmp_dir: str = "",
    validation_file: str = "",
    log_file: str = "validation_logs"
    ):
    
    os.makedirs(tmp_dir, exist_ok=True)
    validation_file, model_output,validated_result, plausible, total = prepare_validation_environment(
        input_file=input_file,output_dir=output_dir,benchmark_name=benchmark_name,model_type=model_type,enhancement_type=enhancement_type,validation_file=validation_file,log_file=log_file)
    
    for k in model_output['data']:
        print('start validating', k)  
        
        if 'output' not in model_output['data'][k]:
            continue
        else:
            expected_count = len(model_output['data'][k]['output'])
        if k in validated_result['data'] and is_validated(validated_result['data'][k], expected_count):
            print('validation result existed!')
            continue
        validated_result['data'][k] = {}
        validated_result['data'][k]['output'] = []
        
        key_list = k.split('_')
        proj, bug_id, bug_line_loc = key_list[0], key_list[1], key_list[-1]
        path = '_'.join(key_list[2: -1])
        if path[0] == '/':
            path = path[1:]
        project_log_file = os.path.join(log_file,f"{proj}_{bug_id}.log")
        function_start_loc, function_end_loc = model_output['data'][k]['function range'].split('-')
        function_start_loc = int(function_start_loc.split(',')[0])
        function_end_loc = int(function_end_loc.split(',')[0])

        total += 1
        current_is_correct = False
        sample_tmp_dir = os.path.join(tmp_dir, f"{proj}_{bug_id}_{uuid4().hex}")
        os.makedirs(sample_tmp_dir, exist_ok=True)

        checkout_defects4j_project(proj, bug_id + 'b', sample_tmp_dir)
        
        buggy_code_file = os.path.join(sample_tmp_dir, path)
        if not os.path.exists(buggy_code_file):
            raise FileNotFoundError(f"[ERROR] Buggy file not found: {buggy_code_file}")
        with open(buggy_code_file, 'r', encoding='utf-8') as f:
            original_code_lines = f.read().splitlines() 

        start_time = time.time()
        init_out, init_err = defects4j_test_suite(sample_tmp_dir)
        standard_time = int(time.time() - start_time)
        baseline_failed_tests = set(parse_test_output(init_out, init_err)['failed_testcases'])
        init_fail_num = len(baseline_failed_tests)
        print(init_fail_num, str(standard_time) + 's')

        trigger, err = defects4j_trigger(sample_tmp_dir)
        triggers = trigger.strip().split('\n')
        for i, trigger in enumerate(triggers):
            triggers[i] = trigger.strip()
        print('trigger number:', len(triggers))
        
        for rank, item in enumerate(model_output['data'][k]['output']):
            patch = item['patch'] if isinstance(item, dict) else item
            fix_code_lines = (
                original_code_lines[:function_start_loc - 1]
                + patch.split('\n')
                + original_code_lines[function_end_loc:]
            )
            with open(buggy_code_file, 'w') as f:
                f.write('\n'.join(fix_code_lines))

            if proj == "Mockito":
                print("Mockito needs separate compilation")
                compile_fix(sample_tmp_dir)
                

            correctness = None
            start_time = time.time()
            failure_reason = None
            out, err = defects4j_test_suite(sample_tmp_dir, timeout=min(300, int(2*standard_time)))
            msg = (str(out)+"\n" + str(err)).lower()
            test_info = parse_test_output(out, err)
            current_failed_tests = set(test_info["failed_testcases"])
            
            new_failed_tests = list(current_failed_tests - baseline_failed_tests)
            fixed_failed_tests = list(baseline_failed_tests - current_failed_tests)
            unchanged_failed_tests = list(current_failed_tests & baseline_failed_tests)
              
            if correctness is None:
                # pass at least one more trigger case
                # have to pass all non-trigger cases
                
                if 'timeout' in msg:
                    correctness = 'timeout'
                    failure_reason = 'Test timeout'
                elif "compilation failed" in msg or "build failure" in msg:
                    correctness = 'uncompilable'
                    failure_reason = 'Compilation Error'
                elif "failing tests: 0" in msg or "all tests passed" in msg:
                    correctness = 'plausible'
                    failure_reason = None
                    if not current_is_correct:
                        plausible += 1
                        current_is_correct = True
                else:
                    correctness = 'wrong'
                    if len(new_failed_tests) > 0:
                        failure_reason = 'Regression'
                    elif len(fixed_failed_tests) > 0 :
                        failure_reason = 'Partially fixed'
                    elif current_failed_tests == baseline_failed_tests: 
                        failure_reason = 'Unchanged failed tests'
            print(plausible, total, rank, f"{correctness.capitalize()} patch:", patch)        
            elapsed_time = time.time() - start_time


            write_log(project_log_file, proj, rank, correctness, elapsed_time, patch, out, err, failure_reason=failure_reason, new_failed_tests=new_failed_tests, exceptions=test_info['exceptions'])
            validated_result['data'][k]['output'].append({
                'rank': rank,
                'patch': patch,
                'correctness': correctness,
                'validation_time': elapsed_time,
                'failure_reason': failure_reason,      
            })
        shutil.rmtree(sample_tmp_dir, ignore_errors=True)
        save_json(validated_result, validation_file)
    return validation_file


if __name__ == '__main__':
    fire.Fire(validate_defects4j)
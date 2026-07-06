import json
import re
import os
import shutil
import time
import textwrap
import subprocess
import fire
from datetime import datetime
from pathlib import Path

LIBS_DIR = "/home/chenshiping/peft4apr/jasper/lib"

def prepare_validation_environment(input_file,output_dir,benchmark_name,model_type,peft_type,train_dataset,validation_file,log_file ):
    
    os.makedirs(log_file, exist_ok=True)
    plausible, total = 0, 0

    if not validation_file:
        validation_file = os.path.join(output_dir,f"{'_'.join(model_type.split('-'))}_{peft_type}_{train_dataset}_on_{benchmark_name}_valid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    if os.path.exists(validation_file):
        print('validation file existed!')
        with open(validation_file, 'r', encoding='utf-8') as f:
            validated_result = json.load(f)

        total = len(validated_result.get('data', {}))
        for _, res_dict in validated_result.get('data', {}).items():
            for item in res_dict.get('output', []):
                if item.get('correctness') == 'plausible':
                    plausible += 1
                    break
    else:
        print('No existing validation file!')
        validated_result = {'patch_file': input_file,'data': {}}
        
    with open(input_file, 'r', encoding='utf-8') as f:
        model_output = json.load(f)

    return validation_file, model_output, validated_result, plausible, total

def write_log(log_file, project, rank, correctness, elapsed_time, patch, test_stdout, test_stderr, failure_reason=None, new_failed_tests=None, exceptions=None):
    log_content=textwrap.dedent(f"""==================================================
                Project: {project}
                Rank: {rank}
                Correctness: {correctness}
                Validation Time: {elapsed_time:.2f} seconds
                Failure reason: {failure_reason if failure_reason else "None"}
                
                PATCH:
                {patch}

                TEST STDOUT:
                {test_stdout}

                TEST STDERR:
                {test_stderr}
                
                NEW FAILED TESTS
                {new_failed_tests if new_failed_tests else "None"}
                
                EXCEPTIONS
                {exceptions if exceptions else "None"}
                
                """)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_content)
        f.write('\n')
    print(f"Logged validation result for {project} at rank {rank} to {log_file}")
        
def exec_command_with_timeout(cmd, timeout=60, cwd=None):
    try:
        result = subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=timeout,cwd=cwd)
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 'TIMEOUT', 'TIMEOUT'

def parse_test_output(out, err):
    combined = str(out) + "\n" + str(err)
    failed_tests = []
    compile_errors = []
    exceptions = []
    tests_run = 0
    failures = 0
    passed = 0
    pass_rate = 0.0

    # ==========================================
    # 1. 编译错误解析 (优先处理)
    # ==========================================
    compile_patterns = [
        r'error:.*',
        r'.*cannot find symbol.*',
        r'.*incompatible types.*',
        r'.*duplicate class:.*',
        r'.*missing return statement.*',
    ]
    for pattern in compile_patterns:
        matches = re.findall(pattern, combined)
        compile_errors.extend(matches)

    # Defects4J Ant javac format
    ant_compile_pattern = r'\[javac\]\s+(.*?\.java:\d+:\s+error:.*)'
    compile_errors.extend(re.findall(ant_compile_pattern, combined))
    compile_errors = list(set([e.strip() for e in compile_errors if e.strip()]))

    # 如果有编译错误，后续的测试解析就不用做了，直接返回
    if compile_errors:
        return {
            'failed_testcases': [],
            'compile_errors': compile_errors,
            'exceptions': [],
            'tests_run': 0,
            'failures': 0,
            'passed': 0,
            'pass_rate': 0.0,
        }

    # ==========================================
    # 2. 测试失败用例具体名称解析
    # ==========================================
    lines = combined.splitlines()
    for line in lines:
        # Defects4J 格式: "- org.jfree.chart..."
        if line.strip().startswith('- '):
            failed_tests.append(line.strip()[2:])

        # 标准 JUnit 格式: "1) testMethod(ClassName)"
        m = re.match(r'\d+\)\s+([^\(]+)\(', line.strip())
        if m:
            failed_tests.append(m.group(1).strip())
            
        # Maven Surefire / Humaneval
        m_surefire = re.search(r'([\w_]+)\(([\w\.]+)\)\s+Time elapsed:.*<<< (FAILURE|ERROR)!', line.strip())
        if m_surefire:
            method_name = m_surefire.group(1)   # test_1
            class_name = m_surefire.group(2)    # humaneval.TEST_FILTER_INTEGERS
            failed_tests.append(f"{class_name}.{method_name}")
            continue
        
    failed_tests = list(set(failed_tests))

    # ==========================================
    # 3. 测试统计数据解析 (多数据集适配关键)
    # ==========================================
    
    # 策略 A: 匹配 Maven / 失败的 JUnitCore (Tests run: 4,  Failures: 1)
    m_maven = re.search(
        r'Tests run:\s*(\d+),\s*Failures:\s*(\d+)(?:,\s*Errors:\s*(\d+))?',
        combined
    )

    # 策略 B: 匹配成功的 JUnitCore (OK (4 tests)) -> 专治 QuixBugs 成功漏报
    m_junit_ok = re.search(r'OK\s*\(\s*(\d+)\s*tests?\s*\)', combined)

    # 策略 C: 匹配 Defects4J 总结格式 (Failing tests: X)
    m_d4j = re.search(r'Failing tests:\s*(\d+)', combined)

    if m_maven:
        tests_run = int(m_maven.group(1))
        failures = int(m_maven.group(2))
        errors = int(m_maven.group(3)) if m_maven.group(3) else 0
        passed = tests_run - failures - errors
        pass_rate = round(passed / tests_run, 3) if tests_run > 0 else 0.0

    elif m_junit_ok:
        # QuixBugs 全过的情况
        tests_run = int(m_junit_ok.group(1))
        failures = 0
        passed = tests_run
        pass_rate = 1.0

    elif m_d4j:
        # Defects4J 的情况
        failures = int(m_d4j.group(1))
        if failures == 0:
            pass_rate = 1.0
            # Defects4J 成功时不知道总数，给个标记值或留 0，但 pass_rate=1.0 确保能被判定为 Plausible
            tests_run = passed = 0 
        else:
            pass_rate = 0.0
            tests_run = passed = 0

    # ==========================================
    # 4. 异常解析
    # ==========================================
    exception_patterns = [
        r'(java\.[\w\.]+Exception)',
        r'(java\.[\w\.]+Error)',
        r'(org\.junit\.[\w\.]+)',
        r'(AssertionError)',
    ]
    for pattern in exception_patterns:
        matches = re.findall(pattern, combined)
        exceptions.extend(matches)

    exceptions = list(set(exceptions))

    return {
        'failed_testcases': failed_tests,
        'compile_errors': compile_errors,
        'exceptions': exceptions,
        'tests_run': tests_run,
        'failures': failures,
        'passed': passed,
        'pass_rate': pass_rate,
    }

def is_validated(proj_data: dict, expected_patch_count: int) -> bool:

    output = proj_data.get('output', [])
    if not output:
        return False

    if len(output) != expected_patch_count:
        return False

    has_unknown = any(
        item.get('correctness') == 'unknown' 
        for item in output
    )
    if has_unknown:
        return False

    return True

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def save_json(data, validation_file):
    with open(validation_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f'results to file {validation_file}')
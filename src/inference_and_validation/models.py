"""
Describes all the models and their configs, preprocessing of the inputs and outputs processing
"""
import os
import json
import codecs
import subprocess
import time
import torch
import gc
import numpy as np
import functools
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_request import OpenAIRequestHandler
from openai import OpenAI
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaTokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoModel
from unixcoder import UniXcoder
# Cache the result of tokenizer processing to avoid redundant computation
@functools.lru_cache(maxsize=None)
def get_fim_token_ids(tokenizer):
    """
    Returns the special token IDs needed for Fill-in-the-Middle (FIM) formatting.

    Returns:
        Tuple of 5 token IDs:
        (BOS token ID, suffix token ID, prefix token ID, middle token ID, pad token ID)
    """
    if "codellama" in tokenizer.name_or_path:
        return (
            tokenizer.bos_token_id,
            tokenizer.suffix_id,
            tokenizer.prefix_id,
            tokenizer.middle_id,
            0,
        )
    elif "deepseek-coder" in tokenizer.name_or_path:
        return (
            tokenizer.bos_token_id,
            tokenizer.encode("<｜fim▁hole｜>", add_special_tokens=False)[0],
            tokenizer.encode("<｜fim▁begin｜>", add_special_tokens=False)[0],
            tokenizer.encode("<｜fim▁end｜>", add_special_tokens=False)[0],
            tokenizer.encode("<pad>", add_special_tokens=False)[0],
        )
    elif "starcoder" in tokenizer.name_or_path:
        return (
            tokenizer.bos_token_id,
            tokenizer.encode("<fim_suffix>")[0],
            tokenizer.encode("<fim_prefix>")[0],
            tokenizer.encode("<fim_middle>")[0],
            tokenizer.encode("<fim_pad>")[0],
        )
    else:
        print('Unknown FIM tokenizer')
        
    return None


def _bos_token_processing(prefix_token_list, bos_token):
    """
    Adds the BOS (beginning-of-sequence) token to the front of a list of tokens, if provided.
    
    Args:
        prefix_token_list (list): List of special tokens (prefix/suffix IDs).
        bos_token (int or None): BOS token ID to prepend if not None.
    
    Returns:
        Modified list with BOS token prepended if applicable.
    """
    if bos_token is not None:
        prefix_token_list.insert(0, bos_token)
    return prefix_token_list


# Adapted from https://github.com/bigcode-project/Megatron-LM/blob/6c4bf908df8fd86b4977f54bf5b8bd4b521003d1/megatron/data/gpt_dataset.py
def permute(
    sample,
    suffix_tok_id,
    prefix_tok_id,
    middle_tok_id,
    pad_tok_id,
    fim_spm_rate=0.5,
    truncate_or_pad=False,
    bos_token_id=None,
):
    """
    Applies a Fill-in-the-Middle (FIM) permutation to a sample.

    Args:
        sample (tuple): Tuple of (prefix, middle, suffix) as NumPy arrays of token IDs.
        suffix_tok_id, prefix_tok_id, middle_tok_id: Special token IDs for suffix/prefix/middle.
        pad_tok_id (int): Token ID used for padding if truncation/padding is enabled.
        fim_spm_rate (float): Probability to apply SPM (Suffix-Prefix-Middle) permutation. Otherwise uses PSM.
        truncate_or_pad (bool): If True, truncates or pads the suffix to ensure a consistent sequence length.
        bos_token_id (int or None): If set, prepends BOS token to the beginning of the sequence.

    Returns:
        List of token IDs representing the permuted sequence.
    """
    prefix, middle, suffix = sample

    if truncate_or_pad:
        # Calculate new target length including 3 special tokens (prefix, suffix, middle markers)
        new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
        diff = new_length - len(sample)

        if diff > 0:
            # Too long — try to truncate suffix to match expected length
            if suffix.shape[0] <= diff:
                return sample
            suffix = suffix[: suffix.shape[0] - diff]
        elif diff < 0:
            # Too short
            suffix = np.concatenate([suffix, np.full((-1 * diff), pad_tok_id)])

    # Apply SPM permutation (Suffix-Prefix-Middle)
    if np.random.rand() < fim_spm_rate:
        prefix_special_tokens = _bos_token_processing(
            [prefix_tok_id, suffix_tok_id], bos_token_id
        )
        new_sample = np.concatenate(
            [
                prefix_special_tokens, 
                suffix,
                [middle_tok_id],
                prefix,
                middle,
            ]
        )
    # Apply PSM permutation (Prefix-Suffix-Middle)
    else:
        prefix_special_tokens = _bos_token_processing([prefix_tok_id], bos_token_id)
        new_sample = np.concatenate(
            [
                prefix_special_tokens,
                prefix,
                [suffix_tok_id],
                suffix,
                [middle_tok_id],
                middle,
            ]
        )
    
    return list(new_sample)

class Model:
    def __init__(self):
        pass


CodeGenInputConfig = {
    "CODEGEN_COMPLETE_CODEFORM_NOCOMMENT": {
        "model_id": "codegen-350M/2B/6B-multi",
        "input": "buggy function before",
        "patch": "code generated by the model, which will replace the entire buggy function. need extra analysis to figure out where to stop"
    },
    "CODEGEN_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT": {
        "model_id": "codegen-350M/2B/6B-multi",
        "input": "buggy function before",
        "patch": "the buggy function before the buggy lines, with buggy lines start with '// buggy line:'. remove all the other commonts and empty lines in the code"
    }
}

class CodeGen(Model):
    def __init__(self, JAVA_DIR, BENCH_DIR, IS_FINETUNED = False):
        super().__init__()
        self.JAVA_DIR = JAVA_DIR
        self.BENCH_DIR = BENCH_DIR
        self.IS_FINETUNED = IS_FINETUNED

    def command(self, cmd):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = process.communicate()
        if output != b'' or err != b'':
            print(output)
            print(err)
        return output, err

    def get_parsed_input(self, filename, start, end, config, tmp_file):
        os.chdir(self.JAVA_DIR)
        print('java', '-cp', '.:target/classes:lib/*', 'clm.codegen.CodeGenInputParser',
            filename, start, end, config, tmp_file)
        self.command([
            'java', '-cp', '.:target/classes:lib/*', 'clm.codegen.CodeGenInputParser',
            filename, start, end, config, tmp_file
        ])

    def get_input(self, config, output_file, bench_dir,shot):
        example_1 = (
            "input: public static int binarySearch(int[] arr, int l, int r, int x) {\n"
            "    if (r >= l) {\n"
            "output:     int mid = l + (r - l) / 2;\n"
            "        if (arr[mid] == x)\n"
            "            return mid;\n"
            "        if (arr[mid] > x)\n"
            "            return binarySearch(arr, l, mid - 1, x);\n"
            "        return binarySearch(arr, mid + 1, r, x);\n"
            "    }\n"
            "    return -1;\n"
            "}\n"
            )
        example_2 = (
            "input: public static int bitcount(int n) {\n   int count = 0;\n    while (n != 0) { while (n != 0) { <extra_id_0> count++; } }}\n  return count;\n}}.\n"
            "output: <extra_id_0> n = (n & (n - 1)); <extra_id_1>"
        )
        example_3 = (
            "input: public static int sumArray(int[] arr) {\n"
            "int sum = 0;\n"
            "for (int i = 0; i < arr.length; i++) {\n"

            "output:     sum += arr[i];\n"
            "}\n"
            "return sum;\n"
            "}\n"
            )
        if "humaneval" in str(bench_dir):
            loc_fp = codecs.open(bench_dir / 'humaneval_loc.txt', 'r', 'utf-8')
            codegen_input = {'config': config, 'data': {}}
            for line in loc_fp.readlines():
                filename, rem_loc = line.strip().split()
                start, end = rem_loc.split('-')
                end = str(int(end) - 1) if end != start else end
                tmp_file = self.BENCH_DIR / 'tmp' / 'humaneval-java' / 'proj' / 'tmp_codegen.json'
                self.get_parsed_input(
                    str(bench_dir / 'src' / 'main' / 'java' / 'humaneval' / 'buggy' / f'{filename}.java'),
                    start,
                    end,
                    config,
                    str(tmp_file)
                )

                if not tmp_file.exists():
                    print(filename, 'failed.', output_file, 'not found.')
                print(filename, 'succeeded')

                result = json.load(open(tmp_file, 'r'))
                codegen_input['data'][filename] = {
                    'loc': rem_loc,
                    'input': result['input'],
                    'function range': result['function range']
                }
                self.command(['rm', '-rf', str(tmp_file)])
            json.dump(codegen_input, open(output_file, 'w'), indent=2)

        elif "quixbugs" in str(bench_dir):
            loc_fp = codecs.open(bench_dir / 'quixbugs_loc.txt', 'r', 'utf-8')
            codegen_input = {'config': config, 'data': {}}
            for line in loc_fp.readlines():
                filename, rem_loc, add_loc = line.strip().split()
                rem_start = int(rem_loc.split('-')[0])
                add_start = int(add_loc.split('-')[0])
                if add_start < rem_start:#选择行号小的作为bug的行号
                    bug_loc = add_loc
                else:
                    bug_loc = rem_loc
                start, end = bug_loc.split('-')#start, end = rem_loc.split('-')
                end = str(int(end) - 1) if end != start else end
                tmp_file = self.BENCH_DIR / 'tmp' / 'quixbugs' / 'proj' / 'tmp_codegen.json'
                self.get_parsed_input(
                    str(bench_dir / 'java_programs' / f'{filename}.java'),
                    start,
                    end,
                    config,
                    str(tmp_file)
                )

                if not tmp_file.exists():
                    print(filename, 'failed.', output_file, 'not found.')
                print(filename, 'succeeded')

                result = json.load(open(tmp_file, 'r'))
                zero_shot_prompt = result['input']
                few_shot_prompt = (
                    f"Example 1:\n{example_1}"
                    f"Example 2:\n{example_3}"
                    f"Target Task:\ninput: {zero_shot_prompt}\noutput: "
                )
                codegen_input['data'][filename] = {
                    'loc': bug_loc,
                    'input': few_shot_prompt,
                    'function range': result['function range']
                }

                self.command(['rm', '-rf', str(tmp_file)])
            json.dump(codegen_input, open(output_file, 'w'), indent=2)

        elif "defects4j" in str(bench_dir):
            loc_fp = codecs.open(bench_dir / 'defects4j_loc.txt', 'r', 'utf-8')
            codegen_input = {'config': config, 'data': {}}
            for line in loc_fp.readlines():
                proj, bug_id, path, rem_loc, add_loc = line.strip().split()
                start, end = rem_loc.split('-')
                end = str(int(end) - 1) if end != start else end
                tmp_dir = self.BENCH_DIR / 'tmp' / 'defects4j' / 'proj' / f'{proj}_{bug_id}'
                tmp_file = tmp_dir / 'tmp_codegen.json'

                print('defects4j', 'checkout', '-p', proj, '-v', bug_id + 'b', '-w', tmp_dir)
                subprocess.call(['defects4j', 'checkout', '-p', proj, '-v', bug_id + 'b', '-w', str(tmp_dir)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.get_parsed_input(str(tmp_dir / path.lstrip('/')), start, end, config, str(tmp_file))

                if not tmp_file.exists():
                    print(proj, bug_id, 'failed.', tmp_file, 'not found.')
                print(proj, bug_id, 'succeeded')

                result = json.load(open(tmp_file, 'r'))
                if result['input'].strip() == '':
                    print(proj, bug_id, 'failed. all empty.')

                result = json.load(open(tmp_file, 'r'))
                filename = f"{proj}_{bug_id}_{path}_{rem_loc}"
                codegen_input['data'][filename] = {
                    'loc': rem_loc,
                    'input': result['input'],
                    'function range': result['function range']
                }

                self.command(['rm', '-rf', str(tmp_file)])
                self.command(['rm', '-rf', str(tmp_dir)])
                json.dump(codegen_input, open(output_file, 'w'), indent=2)
            self.command(['rm', '-rf', str(self.BENCH_DIR / 'tmp' / 'defects4j' / 'proj')])

        else:
            raise "Not known benchmark"

    def create_output(self, input_file, output_file, tokenizer_dir, model_dir, model_name, max_new_tokens, num_output=10):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True ,torch_dtype=torch.float16).to("cuda")
        
        # Try to load PEFT
        """ try:
            model = PeftModel.from_pretrained(
                model,
                model_dir
            )
            model = model.merge_and_unload()
            print('PEFT model loaded successfully')
        except:
            print('Model loaded') """

        codegen_output = json.load(open(input_file, 'r'))
        codegen_output['model'] = model_name
        start_time = time.time()
        for i, filename in enumerate(codegen_output['data']):
            text = codegen_output['data'][filename]['input']
            print(i + 1, 'generating', filename)

            try:
                input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
                if input_ids.size(1) >= 768:
                    print('input too long:', input_ids.size(1), 'skip')
                    continue

                eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
                generated_ids = model.generate(
                    input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=0.8, num_return_sequences=num_output,
                    pad_token_id=eos_id, eos_token_id=eos_id
                )
                output = []

                if self.IS_FINETUNED is not None:
                    for generated_id in generated_ids:
                        o = tokenizer.decode(generated_id[input_ids.size(1):], skip_special_tokens=True)
                        output.append(o)
                else:
                    for generated_id in generated_ids:
                        output.append(tokenizer.decode(generated_id, skip_special_tokens=True))
            except Exception as e:
                print(f"Can't load the model, unexpected exception occured: {e}")
                output = []
            codegen_output['data'][filename]['output'] = output
            json.dump(codegen_output, open(output_file, 'w'), indent=2)
        total_time = int(time.time() - start_time)
        codegen_output['time'] = total_time
        json.dump(codegen_output, open(output_file, 'w'), indent=2)
    """
    #@staticmethod
    def output_to_patch(output, config):
        if 'FINETUNED' in config:
            return output.strip()

        stop_signals = [
            "Output:", "output:", "Example", "Target Task", 
            "###", "public class", "import ", "@author", 
            "public static void main"
        ]

        lines = output.strip().split('\n')
        patch_lines = []
        stack = []
        started = False

        for line in lines:
            clean_line = line.strip()

            if not started:
                if clean_line.startswith("input:") or "public static" in clean_line:
                    started = True
                    patch_lines.append(clean_line.replace("input: ", "", 1))
                    if '{' in clean_line:
                        for c in clean_line:
                            if c == '{':
                                stack.append('{')
                            elif c == '}':
                                if stack:
                                    stack.pop()
                    continue
                else:
                    continue

            if any(sig in clean_line for sig in stop_signals):
                break

            patch_lines.append(line)
            for c in line:
                if c == '{':
                    stack.append('{')
                elif c == '}':
                    if stack:
                        stack.pop()
            if started and len(stack) == 0:
                break

        return '\n'.join(patch_lines).strip()
    """
    @staticmethod
    def output_to_patch(output, config):
        """
        For CODEGEN non-finetuned: extract complete function body continuation.
        The model generates from the buggy line to the end of the function.
        """
        stop_signals = [
            "Output:", "output:", "Example", "Target Task", "*/","private",
            "###", "public", "import ", "@author", "Solution:", "Explanation:"
        ]
        if 'FINETUNED' in config:
            return output.strip()
        else:
            # Remove comments
            lines = output.strip().split('\n')
            no_comment_lines = [line for line in lines if not line.strip().startswith('//')]
            output = '\n'.join(no_comment_lines)
            
            # Remove content after 'public' keyword (e.g., main method)
            cut_index = len(output)
            for signal in stop_signals:
                idx = output.find(signal)
                if idx != -1:
                    cut_index = min(cut_index, idx)

            output = output[:cut_index]
            return output.strip()
   
    @staticmethod
    def prepare_input(fn_before, fn_bug, fn_fix, fn_after, tokenizer):
        inputs = fn_before
        inputs += "// bug start: \n" + fn_bug + "// bug end \n"
        inputs += fn_after + "// fix: \n" + fn_fix + tokenizer.eos_token
        outputs = fn_fix + tokenizer.eos_token

        inputs = tokenizer.encode(inputs, return_tensors='pt')
        outputs = tokenizer.encode(outputs, return_tensors='pt')

        return {
            'input_ids': inputs,
            'labels': torch.cat([torch.zeros(1, inputs.size(1) - outputs.size(1)).fill_(-100).long(), outputs], dim=1),
            'attention_mask': torch.ones(inputs.size()).long()
        }


CodeT5InputConfig = {
    "CODET5_BASE_CODEFORM_MASKFORM_NOCOMMENT": {
        "model_id": "codet5-small/base/large",
        "input": "entire buggy function, with buggy lines masked by <extra_id_0>",
        "patch": "code generated by the model, which will replace the buggy lines"
    },
    "CODET5_BASE_CODEFORM_COMMENTFORM_NOCOMMENT": {
        "model_id": "codet5-small/base/large",
        "input": "entire buggy function, with comments telling the buggy lines and buggy lines masked by <extra_id_0>",
        "patch": "code generated by the model, which will replace the buggy lines"
    },
}

class CodeT5(Model):
    def __init__(self, JAVA_DIR, BENCH_DIR, IS_FINETUNED = False):
        super().__init__()
        self.JAVA_DIR = JAVA_DIR
        self.BENCH_DIR = BENCH_DIR
        self.IS_FINETUNED = IS_FINETUNED

    def command(self, cmd):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = process.communicate()
        if output != b'' or err != b'':
            print(output)
            print(err)
        return output, err

    def get_parsed_input(self, filename, start, end, config, tmp_file):
        os.chdir(self.JAVA_DIR)
        self.command([
            'java', '-cp', '.:target/classes:lib/*', 'clm.codet5.CodeT5InputParser',
            filename, start, end, config, tmp_file
        ])

    @staticmethod
    def prepare_input(fn_before, fn_bug, fn_fix, fn_after, tokenizer):
        inputs = fn_before
        inputs += "// bug start: \n" + fn_bug + "// bug end \n"
        inputs += fn_after
        outputs = fn_fix + tokenizer.eos_token
        
        inputs = tokenizer.encode(inputs, return_tensors='pt')
        outputs = tokenizer.encode(outputs, return_tensors='pt')

        return {
            'input_ids': inputs,
            'labels': outputs,
            'attention_mask': torch.ones(inputs.size()).long()
        }

    def get_input(self, config, output_file, bench_dir, shot):
        example_1 = (
            "input: public static int binarySearch(int[] arr, int l, int r, int x) \n    if (r >= l) {{\n        int mid = l + (r + l) / 2;\n        if (arr[mid] == x)\n            return mid;\n        if (arr[mid] > x)\n            return binarySearch(arr, l, mid - 1, x);\n        return binarySearch(arr, mid + 1, r, x);\n    }}\n    return -1;\n}}"
            "output: <extra_id_0> int mid = l + (r - l) / 2; <extra_id_1>"
        )
        example_2 = (
            "input: public static int bitcount(int n) {\n   int count = 0;\n    while (n != 0) { while (n != 0) { <extra_id_0> count++; } }}\n  return count;\n}}.\n"
            "output: <extra_id_0> n = (n & (n - 1)); <extra_id_1>"
        )
        example_3 = (
            "input: public static int gcd(int a, int b) {\n    while (<extra_id_0>) {\n        int t = a;\n        a = b;\n        b = t % b;\n    }\n    return a;\n}\n"
            "output: <extra_id_0> b != 0 <extra_id_1>"
        )
        if "humaneval" in str(bench_dir):
            loc_fp = codecs.open(bench_dir / 'humaneval_loc.txt', 'r', 'utf-8')
            codet5_input = {'config': config, 'data': {}}
            for line in loc_fp.readlines():
                filename, rem_loc = line.strip().split()
                start, end = rem_loc.split('-')
                end = str(int(end) - 1) if end != start else end

                tmp_file = self.BENCH_DIR / 'tmp' / 'humaneval-java' / 'proj' / 'tmp_codet5.json'
                self.get_parsed_input(
                    str(bench_dir / 'src' / 'main' / 'java' / 'humaneval' / 'buggy' / f'{filename}.java'),
                    start,
                    end,
                    config,
                    str(tmp_file)
                )

                if not tmp_file.exists():
                    print(filename, 'failed.', tmp_file, 'not found.')
                print(filename, 'succeeded')

                result = json.load(open(tmp_file, 'r'))
                zero_shot_input = result['input']
                few_shot_prompt = (
                    f"Example 1:\n{example_1}"
                    f"Example 2:\n{example_3}"
                    f"Target Task:\nInput: {zero_shot_input}\nOutput: "
                )
                codet5_input['data'][filename] = {
                    'loc': rem_loc,
                    'input': result['input'],
                    'function range': result['function range']
                }
                self.command(['rm', '-rf', str(tmp_file)])
            json.dump(codet5_input, open(output_file, 'w'), indent=2)

        elif "quixbugs" in str(bench_dir):
            loc_fp = codecs.open(bench_dir / 'quixbugs_loc.txt', 'r', 'utf-8')
            codet5_input = {'config': config, 'data': {}}
            for line in loc_fp.readlines():
                filename, rem_loc, add_loc = line.strip().split()
                rem_start = int(rem_loc.split('-')[0])
                add_start = int(add_loc.split('-')[0])
                if add_start < rem_start:#选择行号小的作为bug的行号
                    bug_loc = add_loc
                else:
                    bug_loc = rem_loc
                start, end = bug_loc.split('-')
                end = str(int(end) - 1) if end != start else end

                tmp_file = self.BENCH_DIR / 'tmp' / 'quixbugs' / 'proj' / 'tmp_codet5.json'
                self.get_parsed_input(
                    str(bench_dir / 'java_programs' / f'{filename}.java'),
                    start,
                    end,
                    config,
                    str(tmp_file)
                )

                if not tmp_file.exists():
                    print(filename, 'failed.', tmp_file, 'not found.')
                print(filename, 'succeeded')

                result = json.load(open(tmp_file, 'r'))
                zero_shot_input = result['input']
                few_shot_prompt = (
                    f"{example_1}\n\n"
                    f"{example_3}\n\n"
                    f"{zero_shot_input}"
                )
                codet5_input['data'][filename] = {
                    'loc': bug_loc,
                    'input': zero_shot_input,
                    'function range': result['function range']
                }
                self.command(['rm', '-rf', str(tmp_file)])
            json.dump(codet5_input, open(output_file, 'w'), indent=2)

        elif "defects4j" in str(bench_dir):
            loc_fp = codecs.open(bench_dir / 'defects4j_loc.txt', 'r', 'utf-8')
            codet5_input = {'config': config, 'data': {}}
            for line in loc_fp.readlines():
                proj, bug_id, path, rem_loc, add_loc = line.strip().split()
                start, end = rem_loc.split('-')
                end = str(int(end) - 1) if end != start else end
                tmp_dir = self.BENCH_DIR / 'tmp' / 'defects4j' / 'proj' / f'{proj}_{bug_id}'
                tmp_file = tmp_dir / 'tmp_codet5.json'

                print('defects4j', 'checkout', '-p', proj, '-v', bug_id + 'b', '-w', tmp_dir)
                subprocess.call(['defects4j', 'checkout', '-p', proj, '-v', bug_id + 'b', '-w', str(tmp_dir)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.get_parsed_input(str(tmp_dir / path.lstrip('/')), start, end, config, str(tmp_file))

                if not tmp_file.exists():
                    print(proj, bug_id, 'failed.', tmp_file, 'not found.')
                print(proj, bug_id, 'succeeded')

                result = json.load(open(tmp_file, 'r'))
                if result['input'].strip() == '':
                    print(proj, bug_id, 'failed. all empty.')

                result = json.load(open(tmp_file, 'r'))
                filename = f"{proj}_{bug_id}_{path}_{rem_loc}"
                codet5_input['data'][filename] = {
                    'loc': rem_loc,
                    'input': result['input'],
                    'function range': result['function range']
                }

                self.command(['rm', '-rf', str(tmp_file)])
                self.command(['rm', '-rf', str(tmp_dir)])
                json.dump(codet5_input, open(output_file, 'w'), indent=2)
            self.command(['rm', '-rf', str(self.BENCH_DIR / 'tmp' / 'defects4j' / 'proj')])

        else:
            raise "Bad benchmark specified"

    def create_output(self, input_file, output_file, tokenizer_dir, model_dir, model_name, max_new_tokens, num_output=10):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, trust_remote_code=True).to("cuda")
        
        # Try to load PEFT
        try:
            model = PeftModel.from_pretrained(
                model,
                model_dir
            )
            model = model.merge_and_unload()
            print('PEFT model loaded successfully')
        except:
            print('Model loaded')
        
        codet5_output = json.load(open(input_file, 'r'))
        codet5_output['model'] = model_name
        start_time = time.time()
        for i, filename in enumerate(codet5_output['data']):
            text = codet5_output['data'][filename]['input']
            print(i + 1, 'generating', filename)

            try:
                input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
                eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
                # https://huggingface.co/Salesforce/codet5p-220m/blob/main/tokenizer_config.json
                if input_ids.size(1) >= 512:
                    print('input too long:', input_ids.size(1), 'skip')
                    continue

                #generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, num_beams=num_output, num_return_sequences=num_output, early_stopping=True, pad_token_id=eos_id, eos_token_id=eos_id)
                generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens,do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    num_return_sequences=num_output,
                    pad_token_id=eos_id,
                    eos_token_id=eos_id
                )
                output = []
                for generated_id in generated_ids:
                    generated_tokens = tokenizer.decode(generated_id, skip_special_tokens=True)
                    output.append(generated_tokens)
            except Exception as e:
                print(f"Can't load the model, unexpected exception occured: {e}")
                output = []
            codet5_output['data'][filename]['output'] = output
            json.dump(codet5_output, open(output_file, 'w'), indent=2)
        total_time = int(time.time() - start_time)
        codet5_output['time'] = total_time
        json.dump(codet5_output, open(output_file, 'w'), indent=2)

    @staticmethod
    def output_to_patch(output, config):
        # 去掉 extra_id token
        for token in ['<extra_id_0>', '<extra_id_1>', '<extra_id_2>']:
            output = output.replace(token, '').strip()
        if '\nfunction' in output:
            output = output.split('\nfunction')[0]
        # 只取第一行有效代码
        for line in output.split('\n'):
            line = line.strip()
            if not line:
                continue
            if not any(c in line for c in [';', '(', ')', '{', '}']):
                continue
            return line
        return output.strip()
  
  
UnixcoderInputConfig = {
    "Unixcoder_BASE_CODEFORM_MASKFORM_NOCOMMENT": {
        "model_id": "microsoft/unixcoder-base",
        "input": "entire buggy function, with buggy lines masked by <mask>",
        "patch": "code generated by the model, which will replace the buggy lines"
    },
    "Unixcoder_BASE_CODEFORM_COMMENTFORM_NOCOMMENT": {
        "model_id": "microsoft/unixcoder-base",
        "input": "entire buggy function, with comments telling the buggy lines and buggy lines masked by <mask>",
        "patch": "code generated by the model, which will replace the buggy lines"
    },
}

class Unixcoder(Model):
    def __init__(self, JAVA_DIR, BENCH_DIR, IS_FINETUNED = False):
        super().__init__()
        self.JAVA_DIR = JAVA_DIR
        self.BENCH_DIR = BENCH_DIR
        self.IS_FINETUNED = IS_FINETUNED

    def command(self, cmd):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = process.communicate()
        if output != b'' or err != b'':
            print(output)
            print(err)
        return output, err

    def get_parsed_input(self, filename, start, end, config, tmp_file):
        os.chdir(self.JAVA_DIR)
        self.command([
            'java', '-cp', '.:target/classes:lib/*', 'clm.unixcoder.UnixcoderInputParser',
            filename, start, end, config, tmp_file
        ])

    @staticmethod
    def prepare_input(fn_before, fn_bug, fn_fix, fn_after, tokenizer,model):
        inputs = fn_before
        inputs += "// bug start: \n" + fn_bug + "// bug end \n"
        inputs += fn_after
        outputs = fn_fix 
        input_tokens = model.tokenize([inputs], max_length=512, mode="<decoder-only>")
        output_tokens = model.tokenize([outputs], max_length=512, mode="<decoder-only>")
        
        return {
            'input_ids': torch.tensor(input_tokens),
            'labels': torch.tensor(output_tokens)
        }

    def get_input(self, config, output_file, bench_dir,shot):
        if "humaneval" in str(bench_dir):
            loc_fp = codecs.open(bench_dir / 'humaneval_loc.txt', 'r', 'utf-8')
            codet5_input = {'config': config, 'data': {}}
            for line in loc_fp.readlines():
                filename, rem_loc = line.strip().split()
                start, end = rem_loc.split('-')
                end = str(int(end) - 1) if end != start else end

                tmp_file = self.BENCH_DIR / 'tmp' / 'humaneval-java' / 'proj' / 'tmp_codet5.json'
                self.get_parsed_input(
                    str(bench_dir / 'src' / 'main' / 'java' / 'humaneval' / 'buggy' / f'{filename}.java'),
                    start,
                    end,
                    config,
                    str(tmp_file)
                )

                if not tmp_file.exists():
                    print(filename, 'failed.', tmp_file, 'not found.')
                print(filename, 'succeeded')

                result = json.load(open(tmp_file, 'r'))
                codet5_input['data'][filename] = {
                    'loc': rem_loc,
                    'input': result['input'],
                    'function range': result['function range']
                }
                self.command(['rm', '-rf', str(tmp_file)])
            json.dump(codet5_input, open(output_file, 'w'), indent=2)

        elif "quixbugs" in str(bench_dir):
            loc_fp = codecs.open(bench_dir / 'quixbugs_loc.txt', 'r', 'utf-8')
            unixcoder_input = {'config': config, 'data': {}}
            for line in loc_fp.readlines():
                filename, rem_loc, add_loc = line.strip().split()
                start, end = rem_loc.split('-')
                end = str(int(end) - 1) if end != start else end

                tmp_file = self.BENCH_DIR / 'tmp' / 'quixbugs' / 'proj' / 'tmp_unixcoder.json'
                self.get_parsed_input(
                    str(bench_dir / 'java_programs' / f'{filename}.java'),
                    start,
                    end,
                    config,
                    str(tmp_file)
                )

                if not tmp_file.exists():
                    print(filename, 'failed.', tmp_file, 'not found.')
                print(filename, 'succeeded')

                result = json.load(open(tmp_file, 'r'))
                unixcoder_input['data'][filename] = {
                    'loc': rem_loc,
                    'input': result['input'],
                    'function range': result['function range']
                }
                self.command(['rm', '-rf', str(tmp_file)])
            json.dump(unixcoder_input, open(output_file, 'w'), indent=2)
        elif "defects4j" in str(bench_dir):
            loc_fp = codecs.open(bench_dir / 'defects4j_loc.txt', 'r', 'utf-8')
            codet5_input = {'config': config, 'data': {}}
            for line in loc_fp.readlines():
                proj, bug_id, path, rem_loc, add_loc = line.strip().split()
                start, end = rem_loc.split('-')
                end = str(int(end) - 1) if end != start else end
                tmp_dir = self.BENCH_DIR / 'tmp' / 'defects4j' / 'proj' / f'{proj}_{bug_id}'
                tmp_file = tmp_dir / 'tmp_codet5.json'

                print('defects4j', 'checkout', '-p', proj, '-v', bug_id + 'b', '-w', tmp_dir)
                subprocess.call(['defects4j', 'checkout', '-p', proj, '-v', bug_id + 'b', '-w', str(tmp_dir)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.get_parsed_input(str(tmp_dir / path), start, end, config, str(tmp_file))

                if not tmp_file.exists():
                    print(proj, bug_id, 'failed.', tmp_file, 'not found.')
                print(proj, bug_id, 'succeeded')

                result = json.load(open(tmp_file, 'r'))
                if result['input'].strip() == '':
                    print(proj, bug_id, 'failed. all empty.')

                result = json.load(open(tmp_file, 'r'))
                filename = f"{proj}_{bug_id}_{path}_{rem_loc}"
                codet5_input['data'][filename] = {
                    'loc': rem_loc,
                    'input': result['input'],
                    'function range': result['function range']
                }

                self.command(['rm', '-rf', str(tmp_file)])
                self.command(['rm', '-rf', str(tmp_dir)])
                json.dump(codet5_input, open(output_file, 'w'), indent=2)
            self.command(['rm', '-rf', str(self.BENCH_DIR / 'tmp' / 'defects4j' / 'proj')])

        else:
            raise "Bad benchmark specified"

    def create_output(self, input_file, output_file, tokenizer_dir, model_dir, model_name, max_new_tokens, num_output=10):
        #tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        model = UniXcoder(model_dir).to("cuda")
        
        # Try to load PEFT
        try:
            model = PeftModel.from_pretrained(
                model,
                model_dir
            )
            model = model.merge_and_unload()
            print('PEFT model loaded successfully')
        except:
            print('Model loaded')
        
        unixcoder_output = json.load(open(input_file, 'r'))
        unixcoder_output['model'] = model_name
        start_time = time.time()
        for i, filename in enumerate(unixcoder_output['data']):
            text = unixcoder_output['data'][filename]['input']
            print(i + 1, 'generating', filename)

            try:
                #input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
                tokens_ids = model.tokenize([text], max_length=512, mode="<encoder-decoder>")
                source_ids = torch.tensor(tokens_ids).to("cuda")
                #eos_id = model.tokenizer.convert_tokens_to_ids(model.tokenizer.eos_token)
                # https://huggingface.co/Salesforce/codet5p-220m/blob/main/tokenizer_config.json
                if source_ids.size(1) >= 512:
                    print('input too long:', source_ids.size(1), 'skip')
                    continue

                generated_ids = model.generate(
                    source_ids, 
                    decoder_only=False,      
                    beam_size=num_output, 
                    max_length=max_new_tokens
                )
                predictions = model.decode(generated_ids)

                output = [p.replace("<mask0>", "").strip() for p in predictions[0]]
                """
                output = []
                for generated_id in generated_ids:
                    generated_tokens = tokenizer.decode(generated_id, skip_special_tokens=True)
                    output.append(generated_tokens)
                """
            except Exception as e:
                print(f"Can't load the model, unexpected exception occured: {e}")
                output = []
            unixcoder_output['data'][filename]['output'] = output
            json.dump(unixcoder_output, open(output_file, 'w'), indent=2)
        total_time = int(time.time() - start_time)
        unixcoder_output['time'] = total_time
        json.dump(unixcoder_output, open(output_file, 'w'), indent=2)
    @staticmethod
    def output_to_patch(output, config):
        return output.strip()

StarCoderInputConfig = {
    "STARCODER_COMPLETE_CODEFORM_NOCOMMENT": {
        "model_id": "starcoderbase-1B/3B/7B",
        "input": "buggy function before",
        "patch": "code generated by the model, which will replace the entire buggy function. need extra analysis to figure out where to stop"
    },
    "STARCODER_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT": {
        "model_id": "starcoderbase-1B/3B/7B",
        "input": "buggy function before",
        "patch": "the buggy function before the buggy lines, with buggy lines start with '// buggy line:'. remove all the other commonts and empty lines in the code"
    }
}

class StarCoder(Model):
    def __init__(self, JAVA_DIR, BENCH_DIR, IS_FINETUNED = False):
        super().__init__()
        self.JAVA_DIR = JAVA_DIR
        self.BENCH_DIR = BENCH_DIR
        self.IS_FINETUNED = IS_FINETUNED

    def command(self, cmd):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = process.communicate()
        if output != b'' or err != b'':
            print(output)
            print(err)
        return output, err

    def get_parsed_input(self, filename, start, end, config, tmp_file):
        os.chdir(self.JAVA_DIR)
        self.command([
             'java', '-cp', 'target/classes:lib/*', 'clm.starcoder.StarCoderInputParser',
            filename, start, end, config, tmp_file
        ])

    def get_input(self, config, output_file, bench_dir):
        if "humaneval" in str(bench_dir):
            loc_fp = codecs.open(bench_dir / 'humaneval_loc.txt', 'r', 'utf-8')
            starcoder_input = {'config': config, 'data': {}}
            for line in loc_fp.readlines():
                filename, rem_loc = line.strip().split()
                start, end = rem_loc.split('-')
                end = str(int(end) - 1) if end != start else end

                tmp_file = self.BENCH_DIR / 'tmp' / 'humaneval-java' / 'proj' / 'tmp_starcoder.json'
                self.get_parsed_input(
                    bench_dir / 'src' / 'main'/ 'java'/ 'humaneval'/ 'buggy'/ f'{filename}.java',
                    start,
                    end,
                    config,
                    tmp_file
                )

                if not tmp_file.exists():
                    print(filename, 'failed.', tmp_file, 'not found.')
                print(filename, 'succeeded')

                result = json.load(open(tmp_file, 'r'))
                starcoder_input['data'][filename] = {
                    'loc': rem_loc,
                    'input': result['input'],
                    'function range': result['function range']
                }
                self.command(['rm', '-rf', tmp_file])
            json.dump(starcoder_input, open(output_file, 'w'), indent=2)

        elif "quixbugs" in str(bench_dir):
            loc_fp = codecs.open(bench_dir / 'quixbugs_loc.txt', 'r', 'utf-8')
            starcoder_input = {'config': config, 'data': {}}
            for line in loc_fp.readlines():
                filename, rem_loc, add_loc = line.strip().split()
                start, end = rem_loc.split('-')
                end = str(int(end) - 1) if end != start else end

                tmp_file = self.BENCH_DIR / 'tmp' / 'quixbugs' / 'proj' / 'tmp_starcoder.json'
                self.get_parsed_input(
                    str(bench_dir / 'java_programs' / f'{filename}.java'),
                    start,
                    end,
                    config,
                    str(tmp_file)
                )

                if not tmp_file.exists():
                    print(filename, 'failed.', tmp_file, 'not found.')
                print(filename, 'succeeded')

                result = json.load(open(tmp_file, 'r'))
                starcoder_input['data'][filename] = {
                    'loc': rem_loc,
                    'input': result['input'],
                    'function range': result['function range']
                }
                self.command(['rm', '-rf', tmp_file])
            json.dump(starcoder_input, open(output_file, 'w'), indent=2)

        elif "defects4j" in str(bench_dir):
            loc_fp = codecs.open(bench_dir / 'defects4j_loc.txt', 'r', 'utf-8')
            starcoder_input = {'config': config, 'data': {}}
            for line in loc_fp.readlines():
                proj, bug_id, path, rem_loc, add_loc = line.strip().split()
                start, end = rem_loc.split('-')
                end = str(int(end) - 1) if end != start else end
                tmp_dir = self.BENCH_DIR / 'tmp' / 'defects4j' / 'proj' / f'{proj}_{bug_id}'
                tmp_file = tmp_dir / 'tmp_starcoder.json'

                subprocess.call(['defects4j', 'checkout', '-p', proj, '-v', bug_id + 'b', '-w', tmp_dir],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.get_parsed_input(str(tmp_dir / path), start, end, config, str(tmp_file))

                if not tmp_file.exists():
                    print(proj, bug_id, 'failed.', tmp_file, 'not found.')
                print(proj, bug_id, 'succeeded')

                result = json.load(open(tmp_file, 'r'))
                if result['input'].strip() == '':
                    print(proj, bug_id, 'failed. all empty.')

                result = json.load(open(tmp_file, 'r'))
                filename = f"{proj}_{bug_id}_{path}_{rem_loc}"
                starcoder_input['data'][filename] = {
                    'loc': rem_loc,
                    'input': result['input'],
                    'function range': result['function range']
                }

                self.command(['rm', '-rf', tmp_file])
                self.command(['rm', '-rf', tmp_dir])
                json.dump(starcoder_input, open(output_file, 'w'), indent=2)
            self.command(['rm', '-rf', self.BENCH_DIR + f'tmp/defects4j/proj/'])

        else:
            raise "Bad benchmark specified"

    def create_output(self, input_file, output_file, tokenizer_dir, model_dir, model_name, max_new_tokens, num_output=10):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).to("cuda")
        
        # Try to load PEFT
        try:
            model = PeftModel.from_pretrained(
                model,
                model_dir
            )
            model = model.merge_and_unload()
            print('PEFT model loaded successfully')
        except:
            print('Model loaded')
        
        starcoder_output = json.load(open(input_file, 'r'))
        starcoder_output['model'] = model_name
        start_time = time.time()
        for i, filename in enumerate(starcoder_output['data']):
            text = starcoder_output['data'][filename]['input']
            print(i + 1, 'generating', filename)

            try:
                inputs = tokenizer(text, return_tensors="pt").to("cuda")
                # Original was 512
                if inputs['input_ids'].size(1) >= int(tokenizer.model_max_length):
                    print('input too long:', inputs['input_ids'].size(1), 'skip')
                    continue

                eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
                generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_output, num_return_sequences=num_output, early_stopping=True, eos_token_id=eos_id)
                output = []
                for generated_id in generated_ids:
                    generated_tokens = tokenizer.decode(generated_id, skip_special_tokens=False)
                    idx = generated_tokens.find("<fim_middle>")
                    generated_tokens = generated_tokens.replace("<|endoftext|>", "")
                    output.append(generated_tokens[idx + len("<fim_middle>"):])
            except Exception as e:
                print(f"Can't load the model, unexpected exception occured: {e}")
                output = []
            starcoder_output['data'][filename]['output'] = output
            json.dump(starcoder_output, open(output_file, 'w'), indent=2)
        total_time = int(time.time() - start_time)
        starcoder_output['time'] = total_time
        json.dump(starcoder_output, open(output_file, 'w'), indent=2)

    @staticmethod
    def output_to_patch(output, config):
        return output.strip()

    @staticmethod
    def prepare_input(fn_before, fn_bug, fn_fix, fn_after, tokenizer):
        (bos_token_id, suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id) = get_fim_token_ids(tokenizer)
        
        # Update bug info
        fn_bug = "// bug start: \n" + fn_bug + "// bug end \n"

        # Encode
        prefix = tokenizer.encode(fn_before + fn_bug, add_special_tokens=False)
        middle = tokenizer.encode(fn_fix, add_special_tokens=False)
        suffix = tokenizer.encode(fn_after, add_special_tokens=False)

        # Perform fim permuatation
        inputs = permute(
            [prefix, middle, suffix],
            suffix_tok_id,
            prefix_tok_id,
            middle_tok_id,
            pad_tok_id,
            fim_spm_rate=0.5,
            truncate_or_pad=False,
            bos_token_id=bos_token_id,
        )
        inputs += [tokenizer.eos_token_id]
        inputs = [inputs]
        
        # Add EOS
        outputs = list(np.concatenate([middle, [tokenizer.eos_token_id]]))
        outputs = [outputs]

        # Convert to tensors
        inputs = torch.LongTensor(inputs)
        outputs = torch.LongTensor(outputs)
        
        return {
            'input_ids': inputs,
            'labels': torch.cat([torch.zeros(1, inputs.size(1) - outputs.size(1)).fill_(-100).long(), outputs], dim=1),
            'attention_mask': torch.ones(inputs.size()).long()
        }


DeepSeekCoderInputConfig = {
    "DEEPSEEKCODER_COMPLETE_CODEFORM_NOCOMMENT": {
        "model_id": "deepseekcoder-base-1.3B/6.7B",
        "input": "buggy function before",
        "patch": "code generated by the model, which will replace the entire buggy function. need extra analysis to figure out where to stop"
    },
    "DEEPSEEKCODER_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT": {
        "model_id": "deepseekcoder-base-1.3B/6.7B",
        "input": "buggy function before",
        "patch": "the buggy function before the buggy lines, with buggy lines start with '// buggy line:'. remove all the other commonts and empty lines in the code"
    }
}

class DeepSeekCoder(Model):
    def __init__(self, JAVA_DIR, BENCH_DIR, IS_FINETUNED = False):
        super().__init__()
        self.JAVA_DIR = JAVA_DIR
        self.BENCH_DIR = BENCH_DIR
        self.IS_FINETUNED = IS_FINETUNED

    def command(self, cmd):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = process.communicate()
        if output != b'' or err != b'':
            print(output)
            print(err)
        return output, err

    def get_parsed_input(self, filename, start, end, config, tmp_file):
        os.chdir(self.JAVA_DIR)
        self.command([
            'java', '-cp', 'target/classes:lib/*', 'clm.deepseekcoder.DeepSeekCoderInputParser',
            filename, start, end, config, tmp_file
        ])

    def get_input(self, config, output_file, bench_dir,shot='zero'):
        if "humaneval" in str(bench_dir):
            loc_fp = codecs.open(bench_dir / 'humaneval_loc.txt', 'r', 'utf-8')
            deepseekcoder_input = {'config': config, 'data': {}}
            for line in loc_fp.readlines():
                filename, rem_loc = line.strip().split()
                start, end = rem_loc.split('-')
                end = str(int(end) - 1) if end != start else end
                tmp_file = self.BENCH_DIR / 'tmp' / 'humaneval-java' / 'proj' / 'tmp_deepseekcoder.json'
                self.get_parsed_input(
                    bench_dir / 'src' / 'main'/ 'java'/ 'humaneval'/ 'buggy'/ f'{filename}.java',
                    start,
                    end,
                    config,
                    tmp_file
                )

                if not tmp_file.exists():
                    print(filename, 'failed.', tmp_file, 'not found.')
                print(filename, 'succeeded')

                result = json.load(open(tmp_file, 'r'))
                deepseekcoder_input['data'][filename] = {
                    'loc': rem_loc,
                    'input': result['input'],
                    'function range': result['function range']
                }
                self.command(['rm', '-rf', tmp_file])
            json.dump(deepseekcoder_input, open(output_file, 'w'), indent=2)

        elif "quixbugs" in str(bench_dir):
            loc_fp = codecs.open(bench_dir / 'quixbugs_loc.txt', 'r', 'utf-8')
            deepseekcoder_input = {'config': config, 'data': {}}
            examples = []
            if shot == 'few':
                examples_path = '/home/chenshiping/peft4apr/src/benchmarks/example.json'
                examples = json.load(open(examples_path))['quixbugs']
            for line in loc_fp.readlines():
                filename, rem_loc, add_loc = line.strip().split()
                rem_start = int(rem_loc.split('-')[0])
                add_start = int(add_loc.split('-')[0])
                bug_loc = add_loc if add_start < rem_start else rem_loc
                start, end = bug_loc.split('-')#start, end = rem_loc.split('-')
                end = str(int(end) - 1) if end != start else end
                tmp_file = self.BENCH_DIR / 'tmp' / 'quixbugs' / 'proj' / 'tmp_deepseekcoder.json'
                self.get_parsed_input(
                    str(bench_dir / 'java_programs' / f'{filename}.java'),
                    start,
                    end,
                    config,
                    str(tmp_file)
                )
                if not tmp_file.exists():
                    print(filename, 'failed.', tmp_file, 'not found.')
                print(filename, 'succeeded')

                result = json.load(open(tmp_file, 'r'))
                zero_shot_input = result['input']
                instruction = "The following function contains a bug. Output only the complete fixed function with NO explanations, NO markdown, NO comments."
                if shot == 'zero':
                    prompt = (
                        f"[INST] {instruction}\n\n"
                        f"{zero_shot_input} [/INST]"
                    )
                else:
                    example_str = ""
                    for ex in examples:
                        example_str += (
                            f"[INST] {instruction}\n\n"
                            f"{ex['input'].strip()} [/INST] {ex['output'].strip()}\n\n"
                        )
                    prompt = (
                        f"{example_str}"
                        f"[INST] {instruction}\n\n"
                        f"{zero_shot_input} [/INST]"
                    )
                deepseekcoder_input['data'][filename] = {
                    'loc': bug_loc,
                    'input': prompt,
                    'function range': result['function range']
                }
                self.command(['rm', '-rf', tmp_file])
            json.dump(deepseekcoder_input, open(output_file, 'w'), indent=2)

        elif "defects4j" in str(bench_dir):
            loc_fp = codecs.open(bench_dir / 'defects4j_loc.txt', 'r', 'utf-8')
            deepseekcoder_input = {'config': config, 'data': {}}
            for line in loc_fp.readlines():
                proj, bug_id, path, rem_loc, add_loc = line.strip().split()
                start, end = rem_loc.split('-')
                end = str(int(end) - 1) if end != start else end
                tmp_dir = self.BENCH_DIR / 'tmp' / 'defects4j' / 'proj' / f'{proj}_{bug_id}'
                tmp_file = tmp_dir / 'tmp_deepseekcoder.json'

                subprocess.call(['defects4j', 'checkout', '-p', proj, '-v', bug_id + 'b', '-w', tmp_dir],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.get_parsed_input(str(tmp_dir / path), start, end, config, str(tmp_file))

                if not tmp_file.exists():
                    print(proj, bug_id, 'failed.', tmp_file, 'not found.')
                print(proj, bug_id, 'succeeded')

                result = json.load(open(tmp_file, 'r'))
                if result['input'].strip() == '':
                    print(proj, bug_id, 'failed. all empty.')

                result = json.load(open(tmp_file, 'r'))
                filename = f"{proj}_{bug_id}_{path}_{rem_loc}"
                deepseekcoder_input['data'][filename] = {
                    'loc': rem_loc,
                    'input': result['input'],
                    'function range': result['function range']
                }

                self.command(['rm', '-rf', tmp_file])
                self.command(['rm', '-rf', tmp_dir])
                json.dump(deepseekcoder_input, open(output_file, 'w'), indent=2)
            self.command(['rm', '-rf', self.BENCH_DIR + f'tmp/defects4j/proj/'])

        else:
            raise "Bad benchmark specified"

    def create_output(self, input_file, output_file, tokenizer_dir, model_dir, model_name, max_new_tokens, num_output=10):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        #model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True,use_flash_attention_2=False).to("cuda")
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True,torch_dtype=torch.float16,attn_implementation="eager",device_map="auto")
        
        # Try to load PEFT
        try:
            model = PeftModel.from_pretrained(
                model,
                model_dir
            )
            model = model.merge_and_unload()
            print('PEFT model loaded successfully')
        except:
            print('Model loaded')
        
        deepseekcoder_output = json.load(open(input_file, 'r'))
        deepseekcoder_output['model'] = model_name
        start_time = time.time()
        for i, filename in enumerate(deepseekcoder_output['data']):
            text = deepseekcoder_output['data'][filename]['input']
            print(i + 1, 'generating', filename)

            try:
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                if inputs['input_ids'].size(1) >= int(tokenizer.model_max_length):
                    print('input too long:', inputs['input_ids'].size(1), 'skip')
                    continue
                print(f"{filename} input tokens: {inputs['input_ids'].size(1)}, max: {tokenizer.model_max_length}")
                print(f"{filename} generate start")
                
                
                eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
                stop_token = tokenizer.encode("### Instruction:", add_special_tokens=False)
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    num_return_sequences=num_output,
                    pad_token_id=eos_id,
                    eos_token_id=eos_id
                )
                torch.cuda.empty_cache()
                output = []
                for generated_id in generated_ids:
                    generated_tokens = tokenizer.decode(generated_id, skip_special_tokens=True)
                    output.append(generated_tokens[len(text):])
            except Exception as e:
                print(f"Can't load the model, unexpected exception occured: {e}")
                output = []
            deepseekcoder_output['data'][filename]['output'] = output
            json.dump(deepseekcoder_output, open(output_file, 'w'), indent=2)
        total_time = int(time.time() - start_time)
        deepseekcoder_output['time'] = total_time
        json.dump(deepseekcoder_output, open(output_file, 'w'), indent=2)

    @staticmethod
    def output_to_patch(output, config):
        # 截断到第一个 ### Instruction 之前
        if '### Instruction:' in output:
            output = output.split('### Instruction:')[0]
        if '### Response:' in output:
            output = output.split('### Response:')[0]
        
        lines = output.strip().split('\n')
        collected = []
        for line in lines:
            line = line.strip()
            if not line:
                if collected:
                    break
                continue
            if line.startswith('```'):
                if collected:
                    break
                continue
            # 注释行含代码，提取冒号后内容
            if line.startswith('//') and not collected:
                if ':' in line:
                    line = line.split(':', 1)[1].strip()
                else:
                    continue
            if not any(c in line for c in [';', '(', ')', '{', '}']):
                if not collected:
                    continue
            collected.append(line)
            if line.endswith(';') or line.endswith('{') or line.endswith('}'):
                break

        return '\n'.join(collected) if collected else output.strip()

    @staticmethod
    def prepare_input(fn_before, fn_bug, fn_fix, fn_after, tokenizer):
        (bos_token_id, suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id) = get_fim_token_ids(tokenizer)
        
        # Update bug info
        fn_bug = "// bug start: \n" + fn_bug + "// bug end \n"

        # Encode
        prefix = tokenizer.encode(fn_before + fn_bug, add_special_tokens=False)
        middle = tokenizer.encode(fn_fix, add_special_tokens=False)
        suffix = tokenizer.encode(fn_after, add_special_tokens=False)

        # Perform fim permuatation
        inputs = permute(
            [prefix, middle, suffix],
            suffix_tok_id,
            prefix_tok_id,
            middle_tok_id,
            pad_tok_id,
            fim_spm_rate=0.5,
            truncate_or_pad=False,
            bos_token_id=bos_token_id,
        )
        inputs += [tokenizer.eos_token_id]
        inputs = [inputs]
        
        # Add EOS
        outputs = list(np.concatenate([middle, [tokenizer.eos_token_id]]))
        outputs = [outputs]

        # Convert to tensors
        inputs = torch.LongTensor(inputs)
        outputs = torch.LongTensor(outputs)
        
        return {
            'input_ids': inputs,
            'labels': torch.cat([torch.zeros(1, inputs.size(1) - outputs.size(1)).fill_(-100).long(), outputs], dim=1),
            'attention_mask': torch.ones(inputs.size()).long()
        }


BloomInputConfig = {
    "BLOOM_COMPLETE_CODEFORM_NOCOMMENT": {
        "model_id": "bloom-base-560m/1.7B/7.1B",
        "input": "buggy function before",
        "patch": "code generated by the model, which will replace the entire buggy function. need extra analysis to figure out where to stop"
    },
    "BLOOM_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT": {
        "model_id": "bloom-base-560m/1.7B/7.1B",
        "input": "buggy function before",
        "patch": "the buggy function before the buggy lines, with buggy lines start with '// buggy line:'. remove all the other commonts and empty lines in the code"
    }
}

class Bloom(Model):
    def __init__(self, JAVA_DIR, BENCH_DIR, IS_FINETUNED=False):
        super().__init__()
        self.JAVA_DIR = JAVA_DIR
        self.BENCH_DIR = BENCH_DIR
        self.IS_FINETUNED = IS_FINETUNED

    def command(self, cmd):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = process.communicate()
        if output != b'' or err != b'':
            print(output)
            print(err)
        return output, err

    def get_parsed_input(self, filename, start, end, config, tmp_file):
        os.chdir(self.JAVA_DIR)
        self.command([
            'java', '-cp', 'target/classes:lib/*', 'clm.bloom.BloomInputParser',
            filename, start, end, config, tmp_file
        ])

    def get_input(self, config, output_file, bench_dir):
        if "humaneval" in str(bench_dir):
            loc_fp = codecs.open(bench_dir / 'humaneval_loc.txt', 'r', 'utf-8')
            bloom_input = {'config': config, 'data': {}}
            for line in loc_fp.readlines():
                filename, rem_loc = line.strip().split()
                start, end = rem_loc.split('-')
                end = str(int(end) - 1) if end != start else end

                tmp_file = self.BENCH_DIR / 'tmp' / 'humaneval-java' / 'proj' / 'tmp_bloom.json'
                self.get_parsed_input(
                    bench_dir / 'src' / 'main'/ 'java'/ 'humaneval'/ 'buggy'/ f'{filename}.java',
                    start,
                    end,
                    config,
                    tmp_file
                )

                if not tmp_file.exists():
                    print(filename, 'failed.', tmp_file, 'not found.')
                print(filename, 'succeeded')

                result = json.load(open(tmp_file, 'r'))
                bloom_input['data'][filename] = {
                    'loc': rem_loc,
                    'input': result['input'],
                    'function range': result['function range']
                }
                self.command(['rm', '-rf', tmp_file])
            json.dump(bloom_input, open(output_file, 'w'), indent=2)

        elif "quixbugs" in str(bench_dir):
            loc_fp = codecs.open(bench_dir / 'quixbugs_loc.txt', 'r', 'utf-8')
            bloom_input = {'config': config, 'data': {}}
            for line in loc_fp.readlines():
                filename, rem_loc, add_loc = line.strip().split()
                start, end = rem_loc.split('-')
                end = str(int(end) - 1) if end != start else end

                tmp_file = self.BENCH_DIR / 'tmp' / 'quixbugs' / 'proj' / 'tmp_bloom.json'
                self.get_parsed_input(
                    str(bench_dir / 'java_programs' / f'{filename}.java'),
                    start,
                    end,
                    config,
                    str(tmp_file)
                )

                if not tmp_file.exists():
                    print(filename, 'failed.', tmp_file, 'not found.')
                print(filename, 'succeeded')

                result = json.load(open(tmp_file, 'r'))
                bloom_input['data'][filename] = {
                    'loc': rem_loc,
                    'input': result['input'],
                    'function range': result['function range']
                }
                self.command(['rm', '-rf', tmp_file])
            json.dump(bloom_input, open(output_file, 'w'), indent=2)

        elif "defects4j" in str(bench_dir):
            loc_fp = codecs.open(bench_dir / 'defects4j_loc.txt', 'r', 'utf-8')
            bloom_input = {'config': config, 'data': {}}
            for line in loc_fp.readlines():
                proj, bug_id, path, rem_loc, add_loc = line.strip().split()
                start, end = rem_loc.split('-')
                end = str(int(end) - 1) if end != start else end
                tmp_dir = self.BENCH_DIR / 'tmp' / 'defects4j' / 'proj' / f'{proj}_{bug_id}'
                tmp_file = tmp_dir / 'tmp_bloom.json'

                subprocess.call(['defects4j', 'checkout', '-p', proj, '-v', bug_id + 'b', '-w', tmp_dir],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.get_parsed_input(str(tmp_dir / path), start, end, config, str(tmp_file))

                if not tmp_file.exists():
                    print(proj, bug_id, 'failed.', tmp_file, 'not found.')
                print(proj, bug_id, 'succeeded')

                result = json.load(open(tmp_file, 'r'))
                if result['input'].strip() == '':
                    print(proj, bug_id, 'failed. all empty.')

                result = json.load(open(tmp_file, 'r'))
                filename = f"{proj}_{bug_id}_{path}_{rem_loc}"
                bloom_input['data'][filename] = {
                    'loc': rem_loc,
                    'input': result['input'],
                    'function range': result['function range']
                }

                self.command(['rm', '-rf', tmp_file])
                self.command(['rm', '-rf', tmp_dir])
                json.dump(bloom_input, open(output_file, 'w'), indent=2)
            self.command(['rm', '-rf', self.BENCH_DIR + f'tmp/defects4j/proj/'])

        else:
            raise "Bad benchmark specified"

    def create_output(self, input_file, output_file, tokenizer_dir, model_dir, model_name, max_new_tokens, num_output=10):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).to("cuda")
        
        # Try to load PEFT
        try:
            model = PeftModel.from_pretrained(
                model,
                model_dir
            )
            model = model.merge_and_unload()
            print('PEFT model loaded successfully')
        except:
            print('Model loaded')
        
        bloom_output = json.load(open(input_file, 'r'))
        bloom_output['model'] = model_name
        start_time = time.time()
        for i, filename in enumerate(bloom_output['data']):
            text = bloom_output['data'][filename]['input']
            print(i + 1, 'generating', filename)

            try:
                inputs = tokenizer(text, return_tensors="pt").to("cuda")
                # Original was 512
                if inputs['input_ids'].size(1) >= int(tokenizer.model_max_length):
                    print('input too long:', inputs['input_ids'].size(1), 'skip')
                    continue

                eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
                generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_output, num_return_sequences=num_output,
                                            early_stopping=True, eos_token_id=eos_id)
                output = []
                if self.IS_FINETUNED is not None:
                    for generated_id in generated_ids:
                        o = tokenizer.decode(generated_id[inputs["input_ids"].size(1):], truncate_before_pattern=[r"\n\n^#", "^'", "\n\n\n"], skip_special_tokens=True)
                        output.append(o)
                else:
                    output.append(tokenizer.decode(generated_id, truncate_before_pattern=[r"\n\n^#", "^'", "\n\n\n"], skip_special_tokens=True))
            except Exception as e:
                print(f"Can't load the model, unexpected exception occured: {e}")
                output = []
            bloom_output['data'][filename]['output'] = output
            json.dump(bloom_output, open(output_file, 'w'), indent=2)
        total_time = int(time.time() - start_time)
        bloom_output['time'] = total_time
        json.dump(bloom_output, open(output_file, 'w'), indent=2)

    @staticmethod
    def output_to_patch(output, config):
        """
        find the } that matches the first { in the output
        """
        if 'FINETUNED' in config:
            return output.strip()
        else:
            output = output.strip().split('\n')
            no_comment_output = [line for line in output if not line.strip().startswith('//')]
            output = '\n'.join(no_comment_output)
            stack = ['{']
            try:
                start_index = output.index('{')
                patch = output[: start_index + 1]
                for c in output[start_index + 1:]:
                    patch += c
                    if c == '}':
                        top = stack.pop()
                        if top != '{':
                            return ''
                        if len(stack) == 0:
                            return patch.strip()
                    elif c == '{':
                        stack.append(c)
                return ''
            except Exception as e:
                return ''

    @staticmethod
    def prepare_input(fn_before, fn_bug, fn_fix, fn_after, tokenizer):
        inputs = fn_before
        inputs += "// bug start: \n" + fn_bug + "// bug end \n"
        inputs += fn_after + "// fix: \n" + fn_fix + tokenizer.eos_token
        outputs = fn_fix + tokenizer.eos_token

        inputs = tokenizer.encode(inputs, return_tensors='pt')
        outputs = tokenizer.encode(outputs, return_tensors='pt')

        return {
            'input_ids': inputs,
            'labels': torch.cat([torch.zeros(1, inputs.size(1) - outputs.size(1)).fill_(-100).long(), outputs], dim=1),
            'attention_mask': torch.ones(inputs.size()).long()
        }


CodeLlamaInputConfig = {
    "CODELLAMA_COMPLETE_CODEFORM_NOCOMMENT": {
        "model_id": "CodeLlama-7b-hf",
        "input": "buggy function before",
        "patch": "code generated by the model, which will replace the entire buggy function. need extra analysis to figure out where to stop"
    },
    "CODELLAMA_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT": {
        "model_id": "CodeLlama-7b-hf",
        "input": "buggy function before",
        "patch": "the buggy function before the buggy lines, with buggy lines start with '// buggy line:'. remove all the other commonts and empty lines in the code"
    }
}

class CodeLlama(Model):
    def __init__(self, JAVA_DIR, BENCH_DIR, IS_FINETUNED=False):
        super().__init__()
        self.JAVA_DIR = JAVA_DIR
        self.BENCH_DIR = BENCH_DIR
        self.IS_FINETUNED = IS_FINETUNED

    def command(self, cmd):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = process.communicate()
        if output != b'' or err != b'':
            print(output)
            print(err)
        return output, err

    def get_parsed_input(self, filename, start, end, config, tmp_file):
        os.chdir(self.JAVA_DIR)
        self.command([
            'java', '-cp', 'target/classes:lib/*',  'clm.codellama.CodeLlamaInputParser',
            filename, start, end, config, tmp_file
        ])

    def get_input(self, config, output_file, bench_dir, shot='zero'):
        if "humaneval" in str(bench_dir):
            loc_fp = codecs.open(bench_dir / 'humaneval_loc.txt', 'r', 'utf-8')
            codellama_input = {'config': config, 'data': {}}
            for line in loc_fp.readlines():
                filename, rem_loc = line.strip().split()
                start, end = rem_loc.split('-')
                end = str(int(end) - 1) if end != start else end

                tmp_file = self.BENCH_DIR / 'tmp' / 'humaneval-java' / 'proj' / 'tmp_codellama.json'
                self.get_parsed_input(
                    bench_dir / 'src' / 'main'/ 'java'/ 'humaneval'/ 'buggy'/ f'{filename}.java',
                    start,
                    end,
                    config,
                    tmp_file
                )
                if not tmp_file.exists():
                    print(filename, 'failed.', tmp_file, 'not found.')
                print(filename, 'succeeded')

                result = json.load(open(tmp_file, 'r'))
                codellama_input['data'][filename] = {
                    'loc': rem_loc,
                    'input': result['input'],
                    'function range': result['function range']
                }
                self.command(['rm', '-rf', tmp_file])
            json.dump(codellama_input, open(output_file, 'w'), indent=2)

        elif "quixbugs" in str(bench_dir):
            loc_fp = codecs.open(bench_dir / 'quixbugs_loc.txt', 'r', 'utf-8')
            codellama_input = {'config': config, 'data': {}}

            examples = []
            if shot == 'few':
                examples_path = '/home/chenshiping/peft4apr/src/benchmarks/example.json'
                examples = json.load(open(examples_path))['quixbugs']

            for line in loc_fp.readlines():
                filename, rem_loc, add_loc = line.strip().split()

                """                 
                if filename in ('BUCKETSORT', 'MERGESORT'):
                    print(filename, 'skipped (parser incompatible)')
                    continue 
                """
                rem_start = int(rem_loc.split('-')[0])
                add_start = int(add_loc.split('-')[0])
                bug_loc = add_loc if add_start < rem_start else rem_loc
                start, end = bug_loc.split('-')
                end = str(int(end) - 1) if end != start else end

                tmp_file = self.BENCH_DIR / 'tmp' / 'quixbugs' / 'proj' / 'tmp_codellama.json'
                self.get_parsed_input(
                    str(bench_dir / 'java_programs' / f'{filename}.java'),
                    start, end, config, str(tmp_file)
                )

                if not tmp_file.exists():
                    print(filename, 'failed.', tmp_file, 'not found.')
                    continue
                print(filename, 'succeeded')

                result = json.load(open(tmp_file, 'r'))
                zero_shot_input = result['input']

                if shot == 'zero':
                    prompt = (
                    f"[INST] The following Java function contains a bug. "
                    f"Output only the fixed line of code at <BUG_HERE>, with NO explanations, NO markdown formatting, NO comments about changes.\n\n"
                    f"{zero_shot_input} [/INST]"
                )
                else:
                    example_str = ""
                    for ex in examples:
                        example_str += (
                        f"[INST] The following Java function contains a bug. "
                        f"Output only the fixed line of code at <BUG_HERE>, with NO explanations, NO markdown formatting, NO comments about changes.\n\n"
                        f"{ex['input']} [/INST] {ex['output']} </s>"
                        f"<s>"
                    )
                    prompt = (
                        f"{example_str}"
                        f"[INST] The following Java function contains a bug. "
                        f"Output only the fixed line of code at <BUG_HERE>, with NO explanations, NO markdown formatting, NO comments about changes.\n\n"
                        f"{zero_shot_input} [/INST]"
                    )
                codellama_input['data'][filename] = {
                    'loc': bug_loc,
                    'input': prompt,
                    'function range': result['function range']
                }
                self.command(['rm', '-rf', str(tmp_file)])

            json.dump(codellama_input, open(output_file, 'w'), indent=2)

        elif "defects4j" in str(bench_dir):
            loc_fp = codecs.open(bench_dir / 'defects4j_loc.txt', 'r', 'utf-8')
            codellama_input = {'config': config, 'data': {}}
            for line in loc_fp.readlines():
                proj, bug_id, path, rem_loc, add_loc = line.strip().split()
                start, end = rem_loc.split('-')
                end = str(int(end) - 1) if end != start else end
                tmp_dir = self.BENCH_DIR / 'tmp' / 'defects4j' / 'proj' / f'{proj}_{bug_id}'
                tmp_file = tmp_dir / 'tmp_codellama.json'

                subprocess.call(['defects4j', 'checkout', '-p', proj, '-v', bug_id + 'b', '-w', tmp_dir],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.get_parsed_input(str(tmp_dir / path.lstrip('/')), start, end, config, str(tmp_file))#

                if not tmp_file.exists():
                    print(proj, bug_id, 'failed.', tmp_file, 'not found.')
                print(proj, bug_id, 'succeeded')

                result = json.load(open(tmp_file, 'r'))
                if result['input'].strip() == '':
                    print(proj, bug_id, 'failed. all empty.')

                result = json.load(open(tmp_file, 'r'))
                filename = f"{proj}_{bug_id}_{path}_{rem_loc}"
                codellama_input['data'][filename] = {
                    'loc': rem_loc,
                    'input': result['input'],
                    'function range': result['function range']
                }

                self.command(['rm', '-rf', tmp_file])
                self.command(['rm', '-rf', tmp_dir])
                json.dump(codellama_input, open(output_file, 'w'), indent=2)
            self.command(['rm', '-rf', str(self.BENCH_DIR / 'tmp' / 'defects4j' / 'proj')])#

        else:
            raise "Bad benchmark specified"

    def create_output(self, input_file, output_file, tokenizer_dir, model_dir, model_name, max_new_tokens, num_output=10):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        print("fill_token:", repr(tokenizer.fill_token))
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True ,torch_dtype=torch.float16).to("cuda")

        """ try:
            model = PeftModel.from_pretrained(
                model,
                model_dir
            )
            model = model.merge_and_unload()
            print('PEFT model loaded successfully')
        except:
            print('Model loaded') """
        
        codellama_output = json.load(open(input_file, 'r'))
        codellama_output['model'] = model_name
        start_time = time.time()
        for i, filename in enumerate(codellama_output['data']):
            text = codellama_output['data'][filename]['input']
            text = text.replace('<FILL_ME>', '<BUG_HERE>')
            codellama_output['data'][filename]['input'] = text
            print(i + 1, 'generating', filename)
            try:
                inputs = tokenizer(text, return_tensors="pt").to("cuda")
                if inputs['input_ids'].size(1) >= int(tokenizer.model_max_length):
                    print('input too long:', inputs['input_ids'].size(1), 'skip')
                    continue

                eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
                generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.8,
                    top_p=0.95, do_sample=True, num_return_sequences=num_output,
                                             pad_token_id=eos_id, eos_token_id=eos_id)
                output = []
                for generated_id in generated_ids:
                    o = tokenizer.decode(generated_id[inputs["input_ids"].size(1):], skip_special_tokens=True)
                    output.append(o)
            except Exception as e:
                print(f"Can't load the model, unexpected exception occured: {e}")

                import traceback
                traceback.print_exc()
                output = []
            codellama_output['data'][filename]['output'] = output
            json.dump(codellama_output, open(output_file, 'w'), indent=2)

        total_time = int(time.time() - start_time)
        codellama_output['time'] = total_time
        json.dump(codellama_output, open(output_file, 'w'), indent=2)

    @staticmethod
    def output_to_patch(output, config):
        lines = output.strip().split('\n')
        collected = []
        for line in lines:
            line = line.strip()
            if not line:
                if collected:
                    break
                continue
            if line.startswith('```'):
                if collected:
                    break
                continue
            if line.startswith('//') and not collected:
                if ':' in line:
                    line = line.split(':', 1)[1].strip()
                else:
                    continue
            if not any(c in line for c in [';', '(', ')', '{', '}']):
                if not collected:
                    continue
            collected.append(line)
            if line.endswith(';') or line.endswith('{') or line.endswith('}'):
                break
        return '\n'.join(collected) if collected else output.strip()

    @staticmethod
    def prepare_input(fn_before, fn_bug, fn_fix, fn_after, tokenizer):
        (bos_token_id, suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id) = get_fim_token_ids(tokenizer)
        
        # Update bug info
        fn_bug = "// bug start: \n" + fn_bug + "// bug end \n"

        # Encode
        prefix = tokenizer.encode(fn_before + fn_bug, add_special_tokens=False)
        middle = tokenizer.encode(fn_fix, add_special_tokens=False)
        suffix = tokenizer.encode(fn_after, add_special_tokens=False)

        # Perform fim permuatation
        inputs = permute(
            [prefix, middle, suffix],
            suffix_tok_id,
            prefix_tok_id,
            middle_tok_id,
            pad_tok_id,
            fim_spm_rate=0.5,
            truncate_or_pad=False,
            bos_token_id=bos_token_id,
        )
        inputs += [tokenizer.eos_token_id]
        inputs = [inputs]
        
        # Add EOS
        outputs = list(np.concatenate([middle, [tokenizer.eos_token_id]]))
        outputs = [outputs]

        # Convert to tensors
        inputs = torch.LongTensor(inputs)
        outputs = torch.LongTensor(outputs)
        
        return {
            'input_ids': inputs,
            'labels': torch.cat([torch.zeros(1, inputs.size(1) - outputs.size(1)).fill_(-100).long(), outputs], dim=1),
            'attention_mask': torch.ones(inputs.size()).long()
        }

class GPT:
    def __init__(self, JAVA_DIR, BENCH_DIR, model_name="gpt-4o-mini"):
        self.JAVA_DIR = JAVA_DIR
        self.BENCH_DIR = BENCH_DIR
        #self.api_handler = OpenAIRequestHandler()
    def command(self, cmd):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = process.communicate()
        if output != b'' or err != b'':
            print(output)
            print(err)
        return output, err

    def get_parsed_input(self, filename, start, end, config, tmp_file):
        os.chdir(self.JAVA_DIR)
        self.command([
            'java', '-cp', 'target/classes:lib/*', 'clm.deepseekcoder.DeepSeekCoderInputParser',
            filename, start, end, config, tmp_file
        ])        
    def get_input(self, config, output_file, bench_dir):
        gpt_input = {"config": config,"data": {}}

        def read_file_lines(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.readlines()

        def build_prompt(code_before, buggy_code, code_after):
            return f"""The following Java code contains a bug.
{code_before}
<buggy_code>
{buggy_code}
</buggy_code>
{code_after}

Replace only the code inside <buggy_code>.Return only the corrected code. Do NOT add comments or extra text.""".strip()

        if "quixbugs" in str(bench_dir).lower():
            loc_fp = codecs.open(bench_dir / "quixbugs_loc.txt", "r", "utf-8")
            for line in loc_fp.readlines():
                filename, rem_loc, add_loc = line.strip().split()
                start, end = rem_loc.split('-')
                end = str(int(end) - 1) if end != start else end
                tmp_file = self.BENCH_DIR / 'tmp' / 'quixbugs' / 'proj' / 'tmp_deepseekcoder.json'
                self.get_parsed_input(
                    str(bench_dir / 'java_programs' / f'{filename}.java'),
                    start,
                    end,
                    "DEEPSEEKCODER_COMPLETE_CODEFORM_NOCOMMENT",
                    str(tmp_file)
                )

                if not tmp_file.exists():
                    print(filename, 'failed.', tmp_file, 'not found.')
                print(filename, 'succeeded')
                result = json.load(open(tmp_file, 'r'))
                fr = result.get('function range', '')
                if '-' in fr:
                    fstart, fend = fr.split('-')
                    func_start = int(fstart.split(',')[0])
                    func_end = int(fend.split(',')[0])
                else:
                    print(f"[WARN] function range missing or malformed for {filename}, using full file")
                    func_start = 1
                    func_end = len(lines)
                    
                file_path = bench_dir / "java_programs" / f"{filename}.java"
                lines = read_file_lines(file_path)
                func_lines = lines[func_start - 1 : func_end]
                code_before = "".join(func_lines[:int(start) - func_start])
                buggy_code = "".join(func_lines[int(start) - func_start : int(end) - func_start + 1])
                code_after = "".join(func_lines[int(end) - func_start + 1 :])

                prompt = build_prompt(
                    code_before=code_before,
                    buggy_code=buggy_code,
                    code_after=code_after,
                )

                gpt_input["data"][filename] = {
                    "loc": rem_loc,
                    "input": prompt,
                    'function range': result['function range']
                }

        elif "humaneval" in str(bench_dir).lower():
            loc_fp = codecs.open(bench_dir / "humaneval_loc.txt", "r", "utf-8")

            for line in loc_fp.readlines():
                filename, rem_loc = line.strip().split()
                start, end = map(int, rem_loc.split("-"))
                file_path = (bench_dir/ "src"/ "main"/ "java"/ "humaneval"/ "buggy"/ f"{filename}.java")
                lines = read_file_lines(file_path)
                code_before = "".join(lines[:start - 1])
                buggy_code = "".join(lines[start - 1:end])
                code_after = "".join(lines[end:])

                prompt = build_prompt(
                    code_before, buggy_code, code_after, rem_loc
                )

                gpt_input["data"][filename] = {
                    "loc": rem_loc,
                    "input": prompt,
                    "function_range": f"{start},{end}"
                }

                print(filename, "FIM input generated")

        elif "defects4j" in str(bench_dir).lower():
            loc_fp = codecs.open(bench_dir / "defects4j_loc.txt", "r", "utf-8")

            for line in loc_fp.readlines():
                proj, bug_id, path, rem_loc, add_loc = line.strip().split()
                start, end = map(int, rem_loc.split("-"))

                file_path = bench_dir / "defects4j_projects" / proj / path
                lines = read_file_lines(file_path)

                code_before = "".join(lines[:start - 1])
                buggy_code = "".join(lines[start - 1:end])
                code_after = "".join(lines[end:])

                prompt = build_prompt(
                    code_before, buggy_code, code_after, rem_loc
                )

                key = f"{proj}_{bug_id}_{path}_{rem_loc}"

                gpt_input["data"][key] = {
                    "loc": rem_loc,
                    "input": prompt,
                    "function_range": f"{start},{end}"
                }

                print(proj, bug_id, "FIM input generated")

        else:
            raise ValueError("Unsupported benchmark")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(gpt_input, f, indent=2)
        print("All FIM inputs written to:", output_file)
    def create_output(self,input_file,output_file,max_new_tokens,num_output=10,temperature=0.8):
        """
        input_file : json produced by get_input
        output_file: json with generated patches
        """
        data = json.load(open(input_file, 'r'))
        data['model'] = self.model_name

        start_time = time.time()

        for i, filename in enumerate(data['data']):
            prompt = data['data'][filename]['input']
            print(f"{i + 1} generating {filename}")

            outputs = []

            for k in range(num_output):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are an expert Java developer. "
                                    "Your task is to fix the bug in the provided code. "
                                    "CRITICAL: Output ONLY the fixed code block wrapped in ```java ... ``` tags. "
                                    "Do not provide any explanations, comments, or extra text outside the code block."
                                )
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=temperature,
                        max_tokens=max_new_tokens
                    )

                    outputs.append(
                        response.choices[0].message.content.strip()
                    )

                except Exception as e:
                    print(f"[API ERROR] {filename}: {e}")

            data['data'][filename]['output'] = outputs

            json.dump(data, open(output_file, 'w'), indent=2)

        data['time'] = int(time.time() - start_time)
        json.dump(data, open(output_file, 'w'), indent=2)


    @staticmethod
    def output_to_patch(output, config=None):

        if not output:
            return ""
 
        output = output.strip()
        
        if output.startswith("```"):
            lines = output.splitlines()
            if len(lines) >= 2:
                lines = lines[1:-1]
            output = "\n".join(lines).strip()
        
        return output

QwenInputConfig = {
    "QWEN_COMPLETE_CODEFORM_NOCOMMENT": {
        "model_id": "Qwen/Qwen2.5-Coder-7B",
        "input": "buggy function before",
        "patch": (
            "code generated by the model, which will replace the entire buggy function. "
            "the output should be pure code only, without explanations or comments. "
            "need extra analysis to figure out where the function ends"
        )
    },

    "QWEN_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT": {
        "model_id": "Qwen/Qwen2.5-Coder-7B",
        "input": "buggy function before",
        "patch": (
            "the buggy function before the buggy lines, with buggy lines start with "
            "'// buggy line:'. remove all the other comments and empty lines in the code. "
            "the generated patch should be code only"
        )
    }
}


# Models dictionary
models_classes = {}
models_classes['codet5p'] = {'model': CodeT5}
models_classes['codegen'] = {'model': CodeGen}
models_classes['starcoder'] = {'model': StarCoder}
models_classes['deepseekcoder'] = {'model': DeepSeekCoder}
models_classes['bloom'] = {'model': Bloom}
models_classes['codellama'] = {'model': CodeLlama}
models_classes['gpt'] = {'model': GPT}
models_classes['unixcoder'] = {'model': Unixcoder}

# Training dictionary
training_classes = {}
training_classes['codet5p'] = {'tokenizer': AutoTokenizer, 'model': AutoModelForSeq2SeqLM, 'task': 'mask'}
training_classes['codegen'] = {'tokenizer': AutoTokenizer, 'model': AutoModelForCausalLM, 'task': 'regressive'}
training_classes['starcoder'] = {'tokenizer': AutoTokenizer, 'model': AutoModelForCausalLM, 'task': 'fim'}
training_classes['deepseekcoder'] = {'tokenizer': AutoTokenizer, 'model': AutoModelForCausalLM, 'task': 'fim'}
training_classes['bloom'] = {'tokenizer': AutoTokenizer, 'model': AutoModelForCausalLM, 'task': 'fim'}
training_classes['codellama'] = {'tokenizer': AutoTokenizer, 'model': AutoModelForCausalLM, 'task': 'fim'}
training_classes['unixcoder'] = {'tokenizer':AutoTokenizer,'model': AutoModel,'task': 'mask-filling'}
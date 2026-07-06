import json
import fire
import os
OUTPUT_TEMP = """
'======{validated_file} results:===='
Model: {model_type}
PEFT Method: {peft_type}  
train_dataset: {train_dataset}  
validation benchmark : {benchmark_name} 
validation result: 
{validation_result}
"""
def cal_result(
    validated_file: str = "llmpeft4apr/results_new/CodeLlama_7b_hf_apr_new_lora_on_humaneval_validation_24_04_04_11_46_17.json",
    model_type: str = 'CodeLlama-7b-hf',
    peft_type: str = 'lora',
    train_dataset: str = 'apr_new',
    benchmark_name: str = 'humaneval',
):

    validated_result = json.load(open(validated_file, 'r'))
    validated_result = validated_result['data']

    validated_result_fp = validated_file.split('.json')[0] + '_result_look.txt'

    res = []

    pass_k_list = [i for i in range(1, 11)]

    for pass_k in pass_k_list:

        total = 0
        plausible_patches = 0

        for proj, proj_data in validated_result.items():

            total += 1

            outputs = proj_data.get('output', [])

            for rank, item in enumerate(outputs):

                if rank >= pass_k:
                    break

                if item['correctness'] == 'plausible':
                    plausible_patches += 1
                    break

        acc = plausible_patches / total if total > 0 else 0

        res.append(
            f'pass@{pass_k}: plausible patches - {plausible_patches}, '
            f'total problems - {total}, correctness percent - {acc:.4f}'
        )

    with open(validated_result_fp, 'w') as f:
        f.write(
            OUTPUT_TEMP.format(
                validated_file=validated_file,
                model_type=model_type,
                peft_type=peft_type,
                train_dataset=train_dataset,
                benchmark_name=benchmark_name,
                validation_result='\n'.join(res)
            )
        )

        
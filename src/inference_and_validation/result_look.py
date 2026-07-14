import json
import fire
import os
OUTPUT_TEMP = """
'======{validated_file} results:===='
Model: {model_type}
enhancement Method: {enhancement_type} 
validation benchmark : {benchmark_name} 
validation result: 
{validation_result}
"""
def cal_result(
    validated_file: str = "llmpeft4apr/results_new/CodeLlama_7b_hf_apr_new_lora_on_humaneval_validation_24_04_04_11_46_17.json",
    model_type: str = 'CodeLlama-7b-hf',
    enhancement_type: str = 'zero-shot',
    benchmark_name: str = 'humaneval',
):

    with open(validated_file, encoding="utf-8") as f:
        validated_result = json.load(f)
    validated_result = validated_result['data']

    base, _ = os.path.splitext(validated_file)
    validated_result_fp = base + "_result_look.txt"

    res = []

    for pass_k in range(1, 11):

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
                enhancement_type=enhancement_type,
                benchmark_name=benchmark_name,
                validation_result='\n'.join(res)
            )
        )
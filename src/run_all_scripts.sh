#!/bin/bash
# timenow=$(date +\%Y\%m\%d_\%H\%M\%S)
timenow=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
models_dir="$PROJECT_ROOT/models"
model_type="gpt-4o" #CodeLlama-Instruct-7b;deepseekcoder-6.7b
enhance_methods="zero-shot" #zero-shot;few-shot;rag;cot
use_bug_type="false" # set to "true" to include bug type info in prompt


echo "Starting generating..."
benchmark_names=( "quixbugs_c1" "humaneval-java_c1" "defects4j_c1")
for benchmark_name in "${benchmark_names[@]}"; do
    base_name=$(echo $benchmark_name | sed 's/_c[12]$//')
    config=$(echo $benchmark_name | grep -o '_c[12]$' | sed 's/_//')
    output_dir="$PROJECT_ROOT/datasets/results/$base_name/$model_type/$enhance_methods/"
    output_file_name="${model_type}_${enhance_methods}_on_${benchmark_name}_output.json" 
    if [[ "$enhance_methods" == "rag" || "$enhance_methods" == "cot" ]]; then
        python "$PROJECT_ROOT/src/inference_and_validation/rag.py" \
        --benchmark_data "$PROJECT_ROOT/datasets/benchmarks/$benchmark_name.json" \
        --benchmark_name $benchmark_name \
        --select TopK \
        --n_example 2 \
        --num_output 10 \
        --enhance_methods "$enhance_methods" \
        --model_type "$model_type" \
        --model_name_or_path "$models_dir/$model_type" \
        --output_dir "$output_dir" \
        --output_file_name "$output_file_name"
    else
        python "$PROJECT_ROOT/src/inference_and_validation/generate_patch.py" \
        --output_file_name $output_file_name \
        --benchmark_data "$PROJECT_ROOT/datasets/benchmarks/$benchmark_name.json" \
        --benchmark_name $benchmark_name \
        --output_dir "$output_dir" \
        --model_type "$model_type" \
        --model_name_or_path "$models_dir/$model_type" \
        --enhance_methods "$enhance_methods" \
        --use_bug_type $use_bug_type \
        --num_output 10 \
        --max_new_tokens 512 \
        --max_seq_len 1024
    fi 
done


echo "Start validation..."
benchmark_names=( "quixbugs_c1" "humaneval-java_c1" "defects4j_c1")
base_tmp_dir="$PROJECT_ROOT/datasets/tmp_benchmark"
mkdir -p $base_tmp_dir 

for benchmark_name in "${benchmark_names[@]}"; do
    base_name=$(echo $benchmark_name | sed 's/_c[12]$//')
    config=$(echo $benchmark_name | grep -o '_c[12]$' | sed 's/_//')
    output_dir="$PROJECT_ROOT/datasets/results/$base_name/$model_type/$enhance_methods/"
    output_file_name="${model_type}_${enhance_methods}_on_${benchmark_name}_output.json"
    validation_file_name="${model_type}_${enhance_methods}_on_${benchmark_name}_valid.json"
    benchmark_dir="$PROJECT_ROOT/datasets/$base_name/"
    tmp_dir=$benchmark_name'_'$timenow
    cp -r $benchmark_dir $base_tmp_dir'/'$tmp_dir
    python "$PROJECT_ROOT/src/inference_and_validation/patch_validation.py" \
    --input_file  $output_dir$output_file_name  \
    --output_dir $output_dir \
    --benchmark_dir $base_tmp_dir'/'$tmp_dir'/' \
    --benchmark_name $base_name \
    --enhance_methods $enhance_methods \
    --model_type $model_type \
    --validation_file $output_dir$validation_file_name \
    --log_file $output_dir${model_type}_${enhance_methods}_on_${benchmark_name}_validation.log

done

echo "Starting analysis..."
benchmark_names=( "quixbugs_c1" "humaneval-java_c1" "defects4j_c1")
for benchmark_name in "${benchmark_names[@]}"; do
    base_name=$(echo $benchmark_name | sed 's/_c[12]$//')
    config=$(echo $benchmark_name | grep -o '_c[12]$' | sed 's/_//')
    output_dir="$PROJECT_ROOT/datasets/results/$base_name/$model_type/$enhance_methods/"
    validation_file="${output_dir}${model_type}_${enhance_methods}_on_${benchmark_name}_valid.json"
    
    echo "Analysing $benchmark_name ..."
    python "$PROJECT_ROOT/src/analysers/analyse_benchmark.py" \
    "$output_dir" \
    "$base_name" \
    "$model_type" \
    "$PROJECT_ROOT/datasets/benchmarks/$benchmark_name.json"
done
#!/bin/bash
# timenow=$(date +\%Y\%m\%d_\%H\%M\%S)
timenow=$(date +%Y%m%d_%H%M%S)
models_dir="/home/chenshiping/models"
# to modify
model_type="gpt-4o" #CodeLlama-Instruct-7b;deepseekcoder-6.7b
enhance_methods="zero-shot" #lora;zero-shot;few-shot;rag;cot

peft_model_weights="/home/chenshiping/peft4apr/datasets/instruction_tuning_dataset/deepseekcoder-1.3b/output/lora/20260526_100249"
train_dataset="lora_16_apr"

# benchmark_names=("quixbugs-java_c1") #when lora only c2
benchmark_names=("quixbugs-java_c2") #when lora only c2
echo "Starting generating..."
for benchmark_name in "${benchmark_names[@]}"; do
    base_name=$(echo $benchmark_name | sed 's/_c[12]$//')
    config=$(echo $benchmark_name | grep -o '_c[12]$' | sed 's/_//')
    output_dir="/home/chenshiping/peft4apr/datasets/results/$base_name/$model_type/$enhance_methods/"
    output_file_name="${model_type}_${enhance_methods}_on_${benchmark_name}_output.json" 
    if [ "$enhance_methods" = "rag" ]; then
        python src/ThinkRepair-main/rag.py \
        --benchmark_data "datasets/benchmarks/$benchmark_name.json" \
        --benchmark_name $benchmark_name \
        --select TopK \
        --n_example 2 \
        --num_output 10 \
        --enhance_methods "$enhance_methods" \
        --model_type "$model_type" \
        --model_name_or_path "$models_dir/$model_type" \
        --output_dir "$output_dir" \
        --output_file_name "$output_file_name"
    elif [ "$enhance_methods" = "lora" ]; then
        python src/benchmarks/inference_and_validation_src/peft_generate_patch.py \
        --output_file_name $output_file_name \
        --train_dataset "$train_dataset" \
        --benchmark_data "datasets/benchmarks/$benchmark_name.json" \
        --benchmark_name $benchmark_name \
        --output_dir "$output_dir" \
        --model_type "$model_type" \
        --model_name_or_path "$models_dir/$model_type" \
        --is_peft True \
        --enhance_methods "$enhance_methods" \
        --peft_model_weights "$peft_model_weights" \
        --num_output 10 \
        --max_new_tokens 256 \
        --max_seq_len 1024
    elif [ "$enhance_methods" = "cot" ]; then
        CUDA_VISIBLE_DEVICES=1,3 python src/ThinkRepair-main/rag.py \
        --benchmark_data "datasets/benchmarks/$benchmark_name.json" \
        --benchmark_name $benchmark_name \
        --select SSelect \
        --n_example 2 \
        --num_output 10 \
        --enhance_methods "$enhance_methods" \
        --model_type "$model_type" \
        --model_name_or_path "$models_dir/$model_type" \
        --output_dir "$output_dir" \
        --output_file_name "$output_file_name"
    else
        CUDA_VISIBLE_DEVICES=2 python src/benchmarks/inference_and_validation_src/peft_generate_patch.py \
        --output_file_name $output_file_name \
        --train_dataset "$train_dataset" \
        --benchmark_data "datasets/benchmarks/$benchmark_name.json" \
        --benchmark_name $benchmark_name \
        --output_dir "$output_dir" \
        --model_type "$model_type" \
        --model_name_or_path "$models_dir/$model_type" \
        --is_peft False \
        --enhance_methods "$enhance_methods" \
        --peft_model_weights "$peft_model_weights" \
        --num_output 10 \
        --max_new_tokens 256 \
        --max_seq_len 1024 
    fi 
done


# benchmark_names=("humaneval_c1" ) #when lora only c2
benchmark_names=("quixbugs-java_c2" "humaneval_c2" "defects4j_c2") #when lora only c2
base_tmp_dir='/home/chenshiping/peft4apr/datasets/tmp_benchmark'
echo "Start validation..."
mkdir -p $base_tmp_dir 
for benchmark_name in "${benchmark_names[@]}"; do
    base_name=$(echo $benchmark_name | sed 's/_c[12]$//')
    config=$(echo $benchmark_name | grep -o '_c[12]$' | sed 's/_//')
    output_dir="/home/chenshiping/peft4apr/datasets/results/$base_name/$model_type/$enhance_methods/"
    output_file_name="${model_type}_${enhance_methods}_on_${benchmark_name}_output.json"
    validation_file_name="${model_type}_${enhance_methods}_on_${benchmark_name}_valid.json"
    if [[ "$base_name" == *"humaneval"* ]]; then
        benchmark_dir="/home/chenshiping/peft4apr/datasets/humaneval-java/"
        tmp_dir=$benchmark_name'_'$timenow
        cp -r $benchmark_dir $base_tmp_dir'/'$tmp_dir'/'
        python src/benchmarks/inference_and_validation_src/peft_patch_validation.py \
        --input_file  $output_dir$output_file_name  \
        --output_dir $output_dir \
        --benchmark_dir $base_tmp_dir'/'$tmp_dir'/' \
        --benchmark_name $base_name \
        --enhance_methods $enhance_methods \
        --model_type $model_type \
        --train_dataset $train_dataset \
        --validation_file $output_dir$validation_file_name \
        --log_file $output_dir${model_type}_${enhance_methods}_on_${benchmark_name}_validation.log
        rm -rf $base_tmp_dir'/'$tmp_dir'/'
    elif [[ "$base_name" == *"quixbugs-java"* ]]; then
        benchmark_dir='/home/chenshiping/peft4apr/datasets/quixbugs/'
        tmp_dir=$benchmark_name'_'$timenow
        cp -r $benchmark_dir $base_tmp_dir'/'$tmp_dir
        python src/benchmarks/inference_and_validation_src/peft_patch_validation.py \
        --input_file  $output_dir$output_file_name  \
        --output_dir $output_dir \
        --benchmark_dir $base_tmp_dir'/'$tmp_dir \
        --benchmark_name $base_name \
        --enhance_methods $enhance_methods \
        --model_type $model_type \
        --train_dataset $train_dataset \
        --validation_file $output_dir$validation_file_name \
        --log_file $output_dir${model_type}_${enhance_methods}_on_${benchmark_name}_validation.log
        rm -rf $base_tmp_dir'/'$tmp_dir
    elif [ "$base_name" = "defects4j" ]; then
        benchmark_dir='/home/chenshiping/peft4apr/datasets/defects4j/'
        tmp_dir=$benchmark_name'_'$timenow
        cp -r $benchmark_dir $base_tmp_dir'/'$tmp_dir
        python src/benchmarks/inference_and_validation_src/peft_patch_validation.py \
        --input_file  $output_dir$output_file_name  \
        --output_dir $output_dir \
        --benchmark_dir $base_tmp_dir'/'$tmp_dir'/' \
        --benchmark_name $base_name \
        --enhance_methods $enhance_methods \
        --model_type $model_type \
        --train_dataset $train_dataset \
        --validation_time '_'$timenow \
        --validation_file $output_dir$validation_file_name \
        --log_file $output_dir${model_type}_${enhance_methods}_on_${benchmark_name}_validation.log
        rm -rf $base_tmp_dir'/'$tmp_dir'/'
    else
        echo "Wrong benchmark name!"
    fi
done


# # ============================================================
# # 4-GPU 并行 RAG 生成
# # ============================================================
# # # source ~/miniconda3/etc/profile.d/conda.sh && conda activate coderepair && cd ~/peft4apr
# benchmark_names=("quixbugs-java_c1" )
# echo "Starting 4-GPU generation..."
# for benchmark_name in "${benchmark_names[@]}"; do
#     base_name=$(echo $benchmark_name | sed 's/_c[12]$//')
#     output_dir="/home/chenshiping/peft4apr/datasets/results/$base_name/$model_type/$enhance_methods/"
#     output_file_name="${model_type}_${enhance_methods}_on_${benchmark_name}_output.json"
#     mkdir -p "$output_dir"
    
#     for gpu in 0 1 2 3; do
#         CUDA_VISIBLE_DEVICES=$gpu nohup python -u src/ThinkRepair-main/rag.py \
#             --benchmark_data "datasets/benchmarks/$benchmark_name.json" \
#             --benchmark_name "$benchmark_name" \
#             --select TopK --n_example 2 --num_output 10 \
#             --enhance_methods "$enhance_methods" --model_type "$model_type" \
#             --model_name_or_path "$models_dir/$model_type" \
#             --output_dir "$output_dir" \
#             --output_file_name "$output_file_name" \
#             --gpu_id $gpu --num_gpus 4 \
#             > /tmp/rag_${model_type}_${benchmark_name}_gpu$gpu.log 2>&1 &
#         echo "  GPU$gpu PID=$!"
#     done
#     wait   # 等4卡全跑完再跑下一个 benchmark

#     # 合并
#     python3 -c "
# import json
# merged = {}
# for g in range(4):
#     f = '${output_dir}${output_file_name%.json}_gpu' + str(g) + '.json'
#     d = json.load(open(f))
#     merged.setdefault('data', {}).update(d.get('data', {}))
# json.dump(merged, open('${output_dir}${output_file_name}', 'w'), indent=2)
# print('Merged', len(merged.get('data',{})), 'bugs -> ${output_dir}${output_file_name}')
# "
# done
# echo "All done."

# echo "Starting analysis..."
# benchmark_names=( "quixbugs-java_c1" "humaneval_c1" "defects4j_c1" "quixbugs-java_c2" "humaneval_c2" "defects4j_c2" ) #when lora only c2
# for benchmark_name in "${benchmark_names[@]}"; do
#     base_name=$(echo $benchmark_name | sed 's/_c[12]$//')
#     config=$(echo $benchmark_name | grep -o '_c[12]$' | sed 's/_//')
#     output_dir="/home/chenshiping/peft4apr/datasets/results/$base_name/$model_type/$enhance_methods/"
#     validation_file="${output_dir}${model_type}_${enhance_methods}_on_${benchmark_name}_valid.json"
    
#     echo "Analysing $benchmark_name ..."
#     python src/analysers/analyse_benchmark.py \
#     "$output_dir" \
#     "$base_name" \
#     "$model_type" \
#     "datasets/benchmarks/$benchmark_name.json"
# done
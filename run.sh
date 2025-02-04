#!/usr/bin/env bash

# 定義函數來處理模型執行
run_tasks() {
    local dataset=$1
    local input_path=$2
    local output_prefix=$3
    local models=("llama3" "mistral" "gemma2" "phi4" "deepseek-r1:8b")
    local results_count=3

    echo "Processing dataset: $dataset"
    for model in "${models[@]}"
    do
        echo "Running tasks for model: $model"
        for i in $(seq 1 $results_count)
        do
            # 動態生成檔名
            output_file="${dataset}/${output_prefix}_result${i}_${model}.json"
            echo $output_file
            echo "Running $model with output: $output_file"
            conda run --live-stream --name talen python main3.py --input_path "$input_path" --model "$model" --output_path "$output_file"

            # 如果命令失敗，記錄錯誤
            if [ $? -ne 0 ]; then
                echo "Error: Task failed for model $model (output: $output_file)" >&2
            fi
        done
    done
}

# 執行三個資料集
run_tasks "MultiFC" "MultiFC/MultiFC_content1.json" "MultiFC1"
run_tasks "MultiFC" "MultiFC/MultiFC_content2.json" "MultiFC2"
run_tasks "MultiFC" "MultiFC/MultiFC_content3.json" "MultiFC3"
run_tasks "MultiFC" "MultiFC/MultiFC_content4.json" "MultiFC4"
run_tasks "MultiFC" "MultiFC/MultiFC_content5.json" "MultiFC5"

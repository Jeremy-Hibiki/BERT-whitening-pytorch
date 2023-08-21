#!/bin/bash

MODELS=(
    "BAAI/bge-base-zh"
    "BAAI/bge-large-zh"
)


for model in "${MODELS[@]}"; do
    model_url="https://huggingface.co/${model}"
    model_name="${model##*/}"
    if [ ! -r "$model_name" ]; then
        echo "Download ${model_name} from ${model_url}"
        git lfs clone "${model_url}"
    else
        echo "${model_name} already exists."
    fi
done

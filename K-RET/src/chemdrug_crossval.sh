#!/bin/bash

base_name="chemdrug_crossval"
label0=0.2
label1=0.201

for test in {1..4}
do

    log_file="./outputs/chemdrug_cross/${base_name}_${test}_${label0}_${label1}.log"
    bin_file="./outputs/chemdrug_cross/${base_name}_${test}_${label0}_${label1}.bin"

    CUDA_VISIBLE_DEVICES='0' python3 -u run_classification_inhibitor.py \
        --pretrained_model_path ./models/pre_trained_model_scibert/output_model.bin \
        --config_path ./models/pre_trained_model_scibert/scibert_scivocab_uncased/config.json \
        --vocab_path ./models/pre_trained_model_scibert/scibert_scivocab_uncased/vocab.txt \
        --train_path "./datasets/chemdrug_cross/train_${test}" \
        --dev_path "./datasets/chemdrug_cross/dev_${test}" \
        --test_path  "./datasets/chemdrug_cross/test_${test}" \
        --epochs_num 30 --batch_size 32 --kg_name "['ChEBI']" \
        --output_model_path "$bin_file" \
        --weight "[$label0, $label1]" | tee "$log_file" &

    wait
done

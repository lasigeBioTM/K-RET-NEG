#!/bin/bash

base_name="chemdrug_weight"

for label0 in $(seq 0.5 0.5 4)
do
    for label1 in $(seq 0.5 0.5 4)
    do

        log_file="./outputs/chemdrug_weight/${base_name}_${label0}_${label1}.log"
        bin_file="./outputs/chemdrug_weight/${base_name}_${label0}_${label1}.bin"

        CUDA_VISIBLE_DEVICES='1,2,3' python3 -u run_classification_inhibitor.py \
            --pretrained_model_path ./models/pre_trained_model_scibert/output_model.bin \
            --config_path ./models/pre_trained_model_scibert/scibert_scivocab_uncased/config.json \
            --vocab_path ./models/pre_trained_model_scibert/scibert_scivocab_uncased/vocab.txt \
            --train_path "./datasets/chemdrug_cross/train_1" \
            --dev_path "./datasets/chemdrug_cross/dev_1" \
            --test_path  "./datasets/chemdrug_cross/test_1" \
            --epochs_num 20 --batch_size 32 --kg_name "['ChEBI']" \
            --output_model_path "$bin_file" \
            --weight "[$label0, $label1]" | tee "$log_file" &

        wait
    done
done


#labels obtained by doing (1 - (df["label"].value_counts().sort_index() / len(df))).values

label0=0.799
label1=0.201


log_file="./outputs/chemdrug_weight/${base_name}_${label0}_${label1}.log"
bin_file="./outputs/chemdrug_weight/${base_name}_${label0}_${label1}.bin"

CUDA_VISIBLE_DEVICES='1,2,3' python3 -u run_classification_inhibitor.py \
    --pretrained_model_path ./models/pre_trained_model_scibert/output_model.bin \
    --config_path ./models/pre_trained_model_scibert/scibert_scivocab_uncased/config.json \
    --vocab_path ./models/pre_trained_model_scibert/scibert_scivocab_uncased/vocab.txt \
    --train_path "./datasets/chemdrug_cross/train_1" \
    --dev_path "./datasets/chemdrug_cross/dev_1" \
    --test_path  "./datasets/chemdrug_cross/test_1" \
    --epochs_num 20 --batch_size 32 --kg_name "['ChEBI']" \
    --output_model_path "$bin_file" \
    --weight "[$label0, $label1]" | tee "$log_file" &

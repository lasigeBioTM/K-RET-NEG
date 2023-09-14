base_name=chemdrug_crossval
label0=0.799
label1=0.201

mkdir -p ./processed_results
mkdir -p ./processed_results/${label0}_${label1}

for i in {1..4}; do

    bin_file="./outputs/chemdrug_cross/${base_name}_${i}_${label0}_${label1}.bin"
    output_file="./processed_results/${label0}_${label1}/${base_name}_${i}_${label0}_${label1}.log"

    CUDA_VISIBLE_DEVICES='5' python3 -u run_classification_inhibitor.py \
        --pretrained_model_path ./models/pre_trained_model_scibert/output_model.bin \
        --config_path ./models/pre_trained_model_scibert/scibert_scivocab_uncased/config.json \
        --vocab_path ./models/pre_trained_model_scibert/scibert_scivocab_uncased/vocab.txt \
        --train_path "./datasets/chemdrug_cross/train_${i}" \
        --dev_path "./datasets/chemdrug_cross/dev_${i}" \
        --test_path "./datasets/chemdrug_cross/test_${i}" \
        --epochs_num 30 --batch_size 32 --kg_name "['ChEBI']" \
        --testing True \
        --to_test_model "$bin_file" \
        | tee "$output_file" &

    wait

    python3 auxiliar/process_results2.py "$output_file" "./datasets/chemdrug_cross/test_${i}" "$output_file"

done

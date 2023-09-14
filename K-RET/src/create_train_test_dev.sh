#!/bin/bash


for test in {1..4}
do
    cd ../data/
    python3 src/train_test_dev.py
    cd ../K-RET
    mv ../data/ChemDrug_Inh_Act/train datasets/chemdrug_cross/train_$test
    mv ../data/ChemDrug_Inh_Act/test datasets/chemdrug_cross/test_$test
    mv ../data/ChemDrug_Inh_Act/dev datasets/chemdrug_cross/dev_$test
done
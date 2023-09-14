# K-RET-NEG

This study aimed to enhance the accuracy and efficiency of identifying negative relations, specifically focusing on inhibitor relations within biomedical texts. It developed a novel text mining system using state-of-the-art K-RET, which allows the integration of biomedical ontologies. Two datasets were utilized, featuring inhibitor relations and their opposite, activator relations. The K-RET system was adapted to identify negative relations more effectively, achieving high accuracy, precision, recall, and F1 scores for the test set.

This repository contains code and instructions to obtain the Inhibitor Model, which is designed for identifying and classifying sentences related to inhibitors and activators in biomedical text. This README provides an overview of the repository structure and how to use the provided scripts.

## Table of Contents
1. [Getting the Data](#getting-the-data)
2. [Model Preparation (K-RET)](#getting-kret)
3. [Final Architecture](#final-architecture)
4. [Data Processing](#data-processing)
5. [Train, dev, test set](obtain-set)
6. [Training the Model](#training-the-model)
7. [Evaluating Model Predictions](#evaluating-model-predictions)

## Getting the Data <a name="getting-the-data"></a>

Inside `data` is possible to find the steps to obtain the data and the correct architecture necessary: [data Folder](https://github.com/PedroSilvest/try2/tree/main/data)

## Model Preparation (K-RET) <a name="getting-K-RET"></a>

To run the code provided, you need to follow these steps:

1. **Obtain the K-RET Algorithm**:
   - Download the K-RET algorithm from its GitHub repository: [K-RET GitHub](https://github.com/lasigeBioTM/K-RET/tree/main).
   - Clone or download the repository to your local machine.

2. **Replace the 'auxiliar' Folder**:
   - Inside the K-RET folder, locate the `auxiliar/` folder (`K-RET/auxiliar`);
   - Replace the `auxiliar` folder from the original K-RET folder with the one provided here.

3. **Place the 'src' Folder**:
   - in `K-RET/src/` is provided the scripts developed for this project;
   - Place the entire `src` folder inside the K-RET folder.

4. **Copy 'run_classification1.py'**:
   - Locate the `run_classification_inhibitor.py` file in the provided code (`K-RET/run_classification_inhibitor.py`). This is a modification of the original `run_classification.py` where you can give weights of each label as arguments.
   - Copy the `run_classification_inhibitor.py` file and place it inside the K-RET folder.

Once you've completed these steps, you should have the necessary files and folder structure set up within the K-RET directory.

## Final Architecture <a name="final-architecture"></a>

The initial configuration should be like this:

- data/
    - ChemProt/
        - chemprot_development/
        - chemprot_sample/
        - chemprot_test_gs/
        - chemprot_train/
    - DrugProt/drugprot-gs-training-development/
        - development/
        - training/
    - src/
- K-RET/
    - [auxiliar/](https://github.com/PedroSilvest/try2/tree/main/K-RET/auxiliar) replace original with provided here.
    - brain/
    - datasets/
    - models/
    - outputs/
    - uer/
    - [src/](https://github.com/PedroSilvest/try2/tree/main/K-RET/src) add this folder to K-RET
    - [run_classification_inhibitor.py](https://github.com/PedroSilvest/try2/blob/main/K-RET/run_classification_inhibitor.py) add this to K-RET
    - other files present in K-RET

## Data Processing <a name="data-processing"></a>

To get the necessary data for the Inhibitor Model, follow these steps:

1. Navigate to the `data` directory: `cd data`

2. Run the following command to generate various JSON files:
   ```
   python3 src/get_data.py
   ```
   - This command will produce the following JSON files:
     - Separate JSON files for each dataset.
     - `final_duplicates.json`, which merges all datasets while preserving possible duplicates.
     - `final_no_duplicates.json`, which removes duplicates from the merged dataset.
     - Two additional files, `all_complete` and `all_ready`, containing all inhibitor/activator sentences:
       - `all_ready` contains only label and text_a.
       - `all_complete` contains the ID of the PubMed article, text, arg1 (entity 1), arg2 (entity 2), and the label of the sentence.

## Train, dev, test set <a name="obtain-set"></a>

1. Navigate to the `K-RET` directory: `cd K-RET`

2. Run the following command to generate 4 different train, dev and test sets:
   ```
   chmod u+x create_train_test_dev.sh; ./create_train_test_dev.sh
   ```
   - `train_test_dev.py`: present in `data/src/`, can be used to split the `final_no_duplicates.json` dataset into train, test, and development sets.
   - `create_train_test_dev.sh`: This script generates four different train, test, and dev datasets from `final_no_duplicates.json`, by calling 4 times `train_test_dev.py`.

## Training the Model <a name="training-the-model"></a>

To train and evaluate the Inhibitor Model, use the scripts in the `K-RET` directory:

1. **To replicate results and check different weights automatically:**

   - Run `./src/chemdrug_weight.sh`, and the results will be saved in the `output/chemdrug_weight/` directory.

2. **To train four different models:**
   ```
   chmod u+x ./src/chemdrug_crossval.sh; ./src/chemdrug_crossval.sh
   ```
   -The `chemdrug_crossval.sh` script allows you to adjust label weights. By default, it uses the following weights obtained from calculations: 0.799 (for the activator label) and 0.201 (for the inhibitor label).
   -The results of the training process will be saved in the `output/chemdrug_crossval/` directory as both a log file and a corresponding binary model file. The naming convention for these files is as follows:

       -Example Model File: `output/chemdrug_crossval/chemdrug_crossval_1_0.799_0.201.bin`
       -Example Log File: `output/chemdrug_crossval/chemdrug_crossval_1_0.799_0.201.log`
    
    -In the file names, the number (e.g., 1) represents the specific train, dev, and test set, while the weights (0.799 and 0.201) indicate the assigned label weights.


## Evaluating Model Predictions <a name="evaluating-model-predictions"></a>

After training the four different models, you can check the predictions made by each model on the test set by running the following script present in `K-RET` directory:
   ```
   chmod u+x ./src/chemdrug_cross_process.sh; ./src/chemdrug_cross_process.sh
   ```
   -This script will use the four different models to evaluate their predictions on the test set they were trained on. The results are saved in `processed_results/`

Feel free to explore the repository further and adapt the provided scripts as needed for your specific use case.

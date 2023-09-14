#Getting dataset for openprompt

#Must run train_test_dev.py first

import pandas as pd

# Specify the file path
train_path = "ChemDrug_Inh_Act/train_complete"
test_path = "ChemDrug_Inh_Act/test_complete"
dev_path = "ChemDrug_Inh_Act/dev_complete"

def open_prompt_data(init_path, final_path):
    df = pd.read_csv(init_path, sep='\t')

    df = df.rename(columns={'text': 'premise'})
    df['label'] = df['label'].replace({'ACTIVATOR': 0, 'INHIBITOR': 1})
    df['arg1'] = '<e1>' + df['arg1'] + '</e1>'
    df['arg2'] = '<e2>' + df['arg2'] + '</e2>'
    df['hypothesis'] = df['arg1'] + " inhibits " + df['arg2']
    selected_columns = ['label', 'premise', 'hypothesis']
    df[selected_columns].to_csv(final_path, index=False, sep='\t')
    print(df.head())

open_prompt_data(train_path, "open_prompt/train")
open_prompt_data(test_path, "open_prompt/test")
open_prompt_data(train_path, "open_prompt/validation")



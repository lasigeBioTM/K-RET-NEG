import pandas as pd
import numpy as np
import sys

df = pd.read_json('ChemDrug_Inh_Act/final_no_duplicates.json')

df_0 = df[df["label"] == "ACTIVATOR"]
df_1 = df[df['label'] == "INHIBITOR"]

print(len(df_0), len(df_1), len(df))

def create_dataframe(n):

    global df_0, df_1

    n_keep_0 = int(n * len(df_0))
    n_keep_1 = int(n * len(df_1))
    # Randomly choose the rows to keep
    keep_idx_0 = np.random.choice(df_0.index, size=n_keep_0, replace=False)
    keep_idx_1 = np.random.choice(df_1.index, size=n_keep_1, replace=False)
    # Create a new DataFrame with the selected rows
    new_df_0 = df_0.loc[keep_idx_0]
    new_df_1 = df_1.loc[keep_idx_1]
    new_df = pd.concat([new_df_0,new_df_1])
    # Remove the selected rows from the original DataFrame
    df_0.drop(keep_idx_0, inplace=True)
    df_1.drop(keep_idx_1, inplace=True)
    return new_df


def write_final(final_df, write_path, write_path2):
    """
    This function writes the final output data to two separate files.

    Parameters:
    final_df (pandas.DataFrame): A pandas dataframe containing the final output data.
    write_path (str): A string representing the path and name of the file 
        to which the 'label' (label) and 'text_a' (text) columns will be written.
    write_path2 (str): A string representing the path and name of the file
        to which the 'id', 'text', 'arg1', 'arg2', and 'label' columns will be written.
    
    Returns:
    None   
    """
    f = open(write_path,"w", encoding="utf-8")
    g = open(write_path2, "w", encoding="utf-8")
    f.write("label\ttext_a\n")
    g.write("id\ttext\targ1\targ2\tlabel\n")
    for ind in range(len(final_df)):
        text = final_df['text'][ind]
        if final_df['label'][ind] == "INHIBITOR":
            f.write("1\t{}\n".format(text))
        elif final_df['label'][ind] == "ACTIVATOR":
            f.write("0\t{}\n".format(text))
        
        arg1, arg2= final_df['arg1'][ind], final_df['arg2'][ind]
        ids, label = int(final_df['id'][ind]), final_df['label'][ind]
        g.write("{}\t{}\t{}\t{}\t{}\n".format(str(ids), text, arg1, arg2, label))

train = create_dataframe(0.6).reset_index().drop('index', axis=1)
test = create_dataframe(0.5).reset_index().drop('index', axis=1)
dev = pd.concat([df_0, df_1]).reset_index().drop('index', axis=1)


write_final(train, 'ChemDrug_Inh_Act/train','ChemDrug_Inh_Act/train_complete')
write_final(test, 'ChemDrug_Inh_Act/test', 'ChemDrug_Inh_Act/test_complete')
write_final(dev, 'ChemDrug_Inh_Act/dev', 'ChemDrug_Inh_Act/dev_complete')


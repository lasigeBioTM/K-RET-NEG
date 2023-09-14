import pandas as pd
import spacy
from tqdm import tqdm

import os


"""Obtain all ihnibitor/activator sentencesfrom Training, Development and Test 
    datasets (DrugProt and ChemProt).
   It will write two files at the end. One with the data ready to be used to
   the deep learning algorithm K-RET and the other with all information for
   each sentence (id, text, arg1, arg2, relation).
   It will also save all dataframes after data have been filtered
"""

nlp = spacy.load("en_core_sci_scibert")


""" LOAD DATASETS """

""" DRUG PROT - Training and Development"""

#Training
df_abs_tr_dr= pd.read_csv('DrugProt/drugprot-gs-training-development/training/drugprot_training_abstracs.tsv', delimiter='\t',
    header=None, names = ["id", "title", "abstract"])
df_ent_tr_dr= pd.read_csv('DrugProt/drugprot-gs-training-development/training/drugprot_training_entities.tsv', delimiter='\t',
    header=None, names = ["id", "ent_number", "type", "init_pos", "final_pos", "name"])
df_rel_tr_dr= pd.read_csv('DrugProt/drugprot-gs-training-development/training/drugprot_training_relations.tsv', delimiter='\t',
    header=None, names = ["id", "relation", "arg1", "arg2"])

#Development
df_abs_dev_dr = pd.read_csv('DrugProt/drugprot-gs-training-development/development/drugprot_development_abstracs.tsv', delimiter='\t',
    header=None, names = ["id", "title", "abstract"])
df_ent_dev_dr = pd.read_csv('DrugProt/drugprot-gs-training-development/development/drugprot_development_entities.tsv', delimiter='\t',
    header=None, names = ["id", "ent_number", "type", "init_pos", "final_pos", "name"])
df_rel_dev_dr = pd.read_csv('DrugProt/drugprot-gs-training-development/development/drugprot_development_relations.tsv', delimiter='\t',
    header=None, names = ["id", "relation", "arg1", "arg2"])


""" CHEM PROT - Training, Development and Test"""

#Training
df_abs_tr_ch = pd.read_csv('ChemProt/ChemProt/chemprot_training/chemprot_training_abstracts.tsv', delimiter='\t',
    header=None, names = ["id", "title", "abstract"])
df_ent_tr_ch = pd.read_csv('ChemProt/ChemProt/chemprot_training/chemprot_training_entities.tsv', delimiter='\t',
    header=None, names = ["id", "ent_number", "type", "init_pos", "final_pos", "name"])
df_rel_tr_ch = pd.read_csv('ChemProt/ChemProt/chemprot_training/chemprot_training_relations.tsv', delimiter='\t',
    header=None, names = ["id", "CPR", "Y/N", "relation", "arg1", "arg2"])

#Development
df_abs_dev_ch = pd.read_csv('ChemProt/ChemProt/chemprot_development/chemprot_development_abstracts.tsv', delimiter='\t',
    header=None, names = ["id", "title", "abstract"])
df_ent_dev_ch = pd.read_csv('ChemProt/ChemProt/chemprot_development/chemprot_development_entities.tsv', delimiter='\t',
    header=None, names = ["id", "ent_number", "type", "init_pos", "final_pos", "name"])
df_rel_dev_ch = pd.read_csv('ChemProt/ChemProt/chemprot_development/chemprot_development_relations.tsv', delimiter='\t',
    header=None, names = ["id", "CPR", "Y/N", "relation", "arg1", "arg2"])

#Test
df_abs_te_ch = pd.read_csv('ChemProt/ChemProt/chemprot_test_gs/chemprot_test_abstracts_gs.tsv', delimiter='\t',
    header=None, names = ["id", "title", "abstract"])
df_ent_te_ch = pd.read_csv('ChemProt/ChemProt/chemprot_test_gs/chemprot_test_entities_gs.tsv', delimiter='\t',
    header=None, names = ["id", "ent_number", "type", "init_pos", "final_pos", "name"])
df_rel_te_ch = pd.read_csv('ChemProt/ChemProt/chemprot_test_gs/chemprot_test_relations_gs.tsv', delimiter='\t',
    header=None, names = ["id", "CPR", "Y/N", "relation", "arg1", "arg2"])


""" CLASS """

class Extraction:

    def __init__(self, id, title, abstract, ent_n, name, pos1, pos2, rel, arg1, arg2):
        self.id = id
        self.abstract = abstract
        self.title = title
        self.entities = {"ent_number": ent_n, "name": name, 'init_pos': pos1, 'final_pos': pos2}
        self.relations = {"relation": rel, "arg1": arg1, "arg2": arg2}
        self.marked = []

    @staticmethod
    def divide_by_sentences(abstract):
        try:
            doc = nlp(abstract)
            return [str(sent) for sent in doc.sents if "<e1>" in str(sent) and "<e2>" in str(sent)]
        except:
            pass

    def mark_abstract(self):
        relations = self.check_positions()
        sentence = self.title + ' ' + self.abstract
        for rel in relations:
            e1_start, e1_end = rel[0][0], rel[0][1]
            e2_start, e2_end = rel[1][0], rel[1][1]
            if e1_start < e2_start:
                #if sentence[e2_start:e2_end+1] != " ": #novo
                marked_sentence = (
                    sentence[:e1_start] +
                    '<e1>' + sentence[e1_start:e1_end] + '</e1> ' +
                    sentence[e1_end: e2_start] +
                    '<e2>' + sentence[e2_start:e2_end] + '</e2> ' +
                    sentence[e2_end:]
                )
                """else: #antigo
                    marked_sentence = (
                        sentence[:e1_start] +
                        '<e1>' + sentence[e1_start:e1_end] + '</e1> ' +
                        sentence[e1_end+1: e2_start] +
                        '<e2>' + sentence[e2_start:e2_end] + '</e2> ' +
                        sentence[e2_end+1:]
                    )"""
            elif e2_start < e1_start:
                #if sentence[e1_start:e1_end+1] != " ":
                marked_sentence = (
                    sentence[:e2_start] +
                    '<e1>' + sentence[e2_start:e2_end] + '</e1> ' +
                    sentence[e2_end: e1_start] +
                    '<e2>' + sentence[e1_start:e1_end] + '</e2> ' +
                    sentence[e1_end:]
                )
                """else:
                    marked_sentence = (
                        sentence[:e2_start] +
                        '<e1>' + sentence[e2_start:e2_end] + '</e1> ' +
                        sentence[e2_end+1: e1_start] +
                        '<e2>' + sentence[e1_start:e1_end] + '</e2> ' +
                        sentence[e1_end + 1:]
                    )"""

            mark = Extraction.divide_by_sentences(marked_sentence)
            if mark:
                self.marked.extend(mark)


    def check_positions(self):
        result = []
        for arg1, arg2 in zip(self.relations['arg1'], self.relations['arg2']):
            arg1_idx = self.entities['ent_number'].index(arg1)
            arg2_idx = self.entities['ent_number'].index(arg2)
            arg1_positions = (self.entities['init_pos'][arg1_idx], self.entities['final_pos'][arg1_idx])
            arg2_positions = (self.entities['init_pos'][arg2_idx], self.entities['final_pos'][arg2_idx])
            result.append((arg1_positions, arg2_positions))
        return result
    
    def get_arg_name(self, arg):
        arg_idx = self.entities['ent_number'].index(arg)
        return self.entities['name'][arg_idx]

""" FUNCTIONS """
def filter_data(df_rel, df_abs, df_ent):
    """
    This function takes three pandas dataframes containing relation data, 
    abstract data, and entity data and filters them based on specific conditions.
    
    Parameters:
    df_rel (pandas.DataFrame): A pandas dataframe containing relation data.
    df_abs (pandas.DataFrame): A pandas dataframe containing abstract data.
    df_ent (pandas.DataFrame): A pandas dataframe containing entity data.
    
    Returns:
    df_inh_act (pandas.DataFrame): A pandas dataframe containing relation data filtered 
        for only 'INHIBITOR' or 'ACTIVATOR' relations.
    df_abs_filter (pandas.DataFrame): A pandas dataframe containing abstract data filtered based on the 'id' 
        column from the filtered df_inh_act dataframe.
    df_ent_filter (pandas.DataFrame): A pandas dataframe containing entity data filtered based on the 'id' 
        column from the filtered df_inh_act dataframe.
    All three dataframes returned have been reset with new index numbers.
    """
    df_inh_act = df_rel[df_rel['relation'].isin(['INHIBITOR', 'ACTIVATOR'])].copy()
    df_inh_act['arg1'] = df_inh_act['arg1'].apply(lambda x: x[5:])
    df_inh_act['arg2'] = df_inh_act['arg2'].apply(lambda x: x[5:])
    #df_inh_act = df_inh_act[df_inh_act['id'] == 17459764]
    #print(df_inh_act)

    df_abs_filter = df_abs[df_abs['id'].isin(df_inh_act['id'])]
    df_ent_filter = df_ent[df_ent['id'].isin(df_inh_act['id'])]

    df_abs_filter = df_abs_filter.reset_index().drop('index', axis=1) #remove new column index
    df_ent_filter = df_ent_filter.reset_index().drop('index', axis=1)
    df_inh_act = df_inh_act.reset_index().drop('index', axis=1)

    return df_inh_act, df_abs_filter, df_ent_filter

def extract_relations(data_id, df_inh_act, df_abs_filter, df_ent_filter):
    """
    This function extracts relations between entities from abstract text. 
    The abstract sentence with the two sentences determined to have a relation 
    in df_inh_act will be added to the global variable final_df.

    Parameters:
    data_id (str): name of the current dataset being processed.
    df_inh_act (pandas.DataFrame): A pandas dataframe containing relation data 
        filtered for only 'INHIBITOR' or 'ACTIVATOR' relations. 
    df_abs_filter (pandas.DataFrame): A pandas dataframe containing abstract data 
        filtered based on the 'id' column from the filtered df_inh_act dataframe.
    df_ent_filter (pandas.DataFrame): A pandas dataframe containing entity data 
        filtered based on the 'id' column from the filtered df_inh_act dataframe.
    
    Returns:
    df_extraxt (pandas.DataFrame): A pandas dataframe containing the relations extracted
        for each sentence.
    """
    global final_df
    df_extract = pd.DataFrame({"id":[], "text":[], "arg1":[], "arg2":[], "label":[]})

    extracted = set()
    for ind in tqdm(range(len(df_abs_filter)), desc= data_id):
        id_ = df_abs_filter.iloc[ind]["id"]
        abstract = df_abs_filter.iloc[ind]["abstract"]
        title = df_abs_filter.iloc[ind]['title']

        sel_rows_ent = df_ent_filter[df_ent_filter['id'] == id_]
        sel_rows_rel = df_inh_act[df_inh_act['id'] == id_]

        ent_number, name = sel_rows_ent['ent_number'].tolist(), sel_rows_ent['name'].tolist()
        init_pos, final_pos = sel_rows_ent['init_pos'].tolist(), sel_rows_ent['final_pos'].tolist()

        relation, arg1, arg2 = sel_rows_rel['relation'].tolist(), sel_rows_rel['arg1'].tolist(), sel_rows_rel['arg2'].tolist()

        x = Extraction(id_, title, abstract, ent_number, name, init_pos, final_pos, relation, arg1, arg2)
        x.mark_abstract()
        extracted.add(x)

        for ind, y in enumerate(x.marked):
            arg1 = x.get_arg_name(x.relations["arg1"][ind])
            arg2 = x.get_arg_name(x.relations["arg2"][ind])
            label = x.relations['relation'][ind]
            new_row = {'id': x.id, 'text': y, 'arg1': arg1, 
                    'arg2': arg2, 'label': label}
            final_df = final_df.append(new_row, ignore_index=True)
            df_extract = df_extract.append(new_row, ignore_index=True)
    return df_extract

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
    f = open(write_path,"w")
    g = open(write_path2, "w")
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

""" FILTER AND RELATION EXTRACTION """

final_df = pd.DataFrame({"id":[], "text":[], "arg1":[], "arg2":[], "label":[]})

all_df =    {
            "DrugProt Train": [df_rel_tr_dr, df_abs_tr_dr, df_ent_tr_dr],
            "DrugProt Dev": [df_rel_dev_dr, df_abs_dev_dr, df_ent_dev_dr],
            "ChemProt Train": [df_rel_tr_ch, df_abs_tr_ch, df_ent_tr_ch],
            "ChemProt Test": [df_rel_te_ch, df_abs_te_ch, df_ent_te_ch],
            "ChemProt Dev": [df_rel_dev_ch, df_abs_dev_ch, df_ent_dev_ch]
            }


""" to save dataframes of inh and act """
directory = r'ChemDrug_Inh_Act/'

if not os.path.exists(directory):
    os.makedirs(directory)


for data_id in all_df:
    df_rel, df_abs, df_ent = all_df[data_id][0],all_df[data_id][1],all_df[data_id][2]
    df_inh_act, df_abs_filter, df_ent_filter = filter_data(df_rel, df_abs, df_ent)
    df_extracted = extract_relations(data_id, df_inh_act, df_abs_filter, df_ent_filter)
    df_extracted.to_json(directory + data_id + ".json")
    
#REMOVE DUPLICATED
final_df.to_json(directory + 'final_with_duplicates.json')
final_df = final_df.drop_duplicates().reset_index().drop('index', axis=1)
final_df.to_json(directory + 'final_no_duplicates.json')

""" WRITE FINAL FILE """
write_final(final_df, directory + "all_ready", directory + "all_complete")

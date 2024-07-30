# from datasets import list_datasets, load_dataset
# dataset = load_dataset('ade_corpus_v2', "Ade_corpus_v2_drug_ade_relation")
# print(dataset)
#
# train = dataset["train"]
# text = train["text"]
# drug = train["drug"]
# effect = train["effect"]
# indexes = train["indexes"]

import pandas as pd

unique_ids = set()
unique_sentences = set()

# Define your column names
column_names = ["id", "Sentence", "ADE", "ade_start", "ade_end", "drug", "drug_start", "drug_end"]
dataset = pd.DataFrame(columns=column_names)

# Replace 'your_file.rel' with the path to your actual file
file_path = 'data/raw/ADE/DRUG-AE.rel'

# Open the file and read its contents
with open(file_path, 'r') as file:
    contents = file.readlines()

for x in contents:
    content = x.split("|")
    data_dict = {"id": content[0].strip(),
                 "Sentence": content[1].strip(),
                 "ADE": content[2].strip(),
                 "ade_start": content[3].strip(),
                 "ade_end": content[4].strip(),
                 "drug": content[5].strip(),
                 "drug_start": content[6].strip(),
                 "drug_end": content[7].strip()
                 }
    dataset = dataset.append(data_dict, ignore_index=True)

dataset.to_csv("data/raw/ADE/dataset.csv")

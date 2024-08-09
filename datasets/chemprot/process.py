import json

import pandas as pd
from tqdm import tqdm
from universal_classes import Entity, Relation, Example, Dataset
from utils import pickle_save



df_abstracts = pd.read_table("chemprot_training/chemprot_training_abstracts.tsv", header=None,
                             keep_default_na=False,
                             names=["doc_key", "title", "abstract"], encoding='utf-8')
df_entities = pd.read_table("chemprot_training/chemprot_training_entities.tsv", header=None,
                            keep_default_na=False,
                            names=["doc_key", "entity_id", "label", "char_start", "char_end", "text"], encoding='utf-8')
df_relations = pd.read_table("chemprot_training/chemprot_training_gold_standard.tsv", header=None,
                             keep_default_na=False,
                             names=["doc_key", "label", "arg1", "arg2"], encoding='utf-8')

dataset = {}
for abs in df_abstracts.iterrows():
    abs = abs[1]
    out_dict = {}
    out_dict['title'] = abs.title
    out_dict['doc_key'] = abs.doc_key
    out_dict['abstract'] = abs.abstract
    out_dict['entities'] = []
    out_dict['relations'] = []
    dataset[abs.doc_key] = out_dict

for ent in df_entities.iterrows():
    ent = ent[1]
    out_dict = dataset[ent.doc_key]
    ent_dict = {}
    ent_dict['text'] = ent.text
    ent_dict['entity_id'] = ent.entity_id
    ent_dict['label'] = ent.label
    out_dict['entities'].append(ent_dict)

for rel in df_relations.iterrows():
    rel = rel[1]
    out_dict = dataset[rel.doc_key]
    entities = out_dict['entities']
    rel_dict = {}
    head = rel.arg1[5:]
    for e in entities:
        if e['entity_id'] == head:
            head = e['text']
    tail = rel.arg2[5:]
    for e in entities:
        if e['entity_id'] == tail:
            tail = e['text']
    rel_dict['head'] = head
    rel_dict['tail'] = tail
    rel_dict['label'] = rel.label
    if rel_dict not in out_dict['relations']:
        out_dict['relations'].append(rel_dict)

lines = [v for _, v in dataset.items()]

with open('train.json', 'w') as file:
    for line in lines:
        file.write(json.dumps(line) + '\n')

df_abstracts = pd.read_table("chemprot_development/chemprot_development_abstracts.tsv", header=None,
                             keep_default_na=False,
                             names=["doc_key", "title", "abstract"], encoding='utf-8')
df_entities = pd.read_table("chemprot_development/chemprot_development_entities.tsv", header=None,
                            keep_default_na=False,
                            names=["doc_key", "entity_id", "label", "char_start", "char_end", "text"], encoding='utf-8')
df_relations = pd.read_table("chemprot_development/chemprot_development_gold_standard.tsv", header=None,
                             keep_default_na=False,
                             names=["doc_key", "label", "arg1", "arg2"], encoding='utf-8')

dataset = {}
for abs in df_abstracts.iterrows():
    abs = abs[1]
    out_dict = {}
    out_dict['title'] = abs.title
    out_dict['doc_key'] = abs.doc_key
    out_dict['abstract'] = abs.abstract
    dataset[abs.doc_key] = out_dict
    out_dict['entities'] = []
    out_dict['relations'] = []

for ent in df_entities.iterrows():
    ent = ent[1]
    out_dict = dataset[ent.doc_key]
    ent_dict = {}
    ent_dict['text'] = ent.text
    ent_dict['entity_id'] = ent.entity_id
    ent_dict['label'] = ent.label
    out_dict['entities'].append(ent_dict)

for rel in df_relations.iterrows():
    rel = rel[1]
    out_dict = dataset[rel.doc_key]
    entities = out_dict['entities']
    rel_dict = {}
    head = rel.arg1[5:]
    for e in entities:
        if e['entity_id'] == head:
            head = e['text']
    tail = rel.arg2[5:]
    for e in entities:
        if e['entity_id'] == tail:
            tail = e['text']
    rel_dict['head'] = head
    rel_dict['tail'] = tail
    rel_dict['label'] = rel.label
    if rel_dict not in out_dict['relations']:
        out_dict['relations'].append(rel_dict)

lines = [v for _, v in dataset.items()]

with open('val.json', 'w') as file:
    for line in lines:
        file.write(json.dumps(line) + '\n')

df_abstracts = pd.read_table("chemprot_test_gs/chemprot_test_abstracts_gs.tsv", header=None,
                             keep_default_na=False,
                             names=["doc_key", "title", "abstract"], encoding='utf-8')
df_entities = pd.read_table("chemprot_test_gs/chemprot_test_entities_gs.tsv", header=None,
                            keep_default_na=False,
                            names=["doc_key", "entity_id", "label", "char_start", "char_end", "text"], encoding='utf-8')
df_relations = pd.read_table("chemprot_test_gs/chemprot_test_gold_standard.tsv", header=None,
                             keep_default_na=False,
                             names=["doc_key", "label", "arg1", "arg2"], encoding='utf-8')

dataset = {}
for abs in df_abstracts.iterrows():
    abs = abs[1]
    out_dict = {}
    out_dict['title'] = abs.title
    out_dict['doc_key'] = abs.doc_key
    out_dict['abstract'] = abs.abstract
    dataset[abs.doc_key] = out_dict
    out_dict['entities'] = []
    out_dict['relations'] = []

for ent in df_entities.iterrows():
    ent = ent[1]
    out_dict = dataset[ent.doc_key]
    ent_dict = {}
    ent_dict['text'] = ent.text
    ent_dict['entity_id'] = ent.entity_id
    ent_dict['label'] = ent.label
    out_dict['entities'].append(ent_dict)

for rel in df_relations.iterrows():
    rel = rel[1]
    out_dict = dataset[rel.doc_key]
    entities = out_dict['entities']
    rel_dict = {}
    head = rel.arg1[5:]
    for e in entities:
        if e['entity_id'] == head:
            head = e['text']
    tail = rel.arg2[5:]
    for e in entities:
        if e['entity_id'] == tail:
            tail = e['text']
    rel_dict['head'] = head
    rel_dict['tail'] = tail
    rel_dict['label'] = rel.label
    if rel_dict not in out_dict['relations']:
        out_dict['relations'].append(rel_dict)

lines = [v for _, v in dataset.items()]

with open('test.json', 'w') as file:
    for line in lines:
        file.write(json.dumps(line) + '\n')

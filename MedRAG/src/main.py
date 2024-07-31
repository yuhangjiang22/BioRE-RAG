#%% imports
# import pydevd_pycharm
import fire
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import universal_classes
import importlib
import json

from itertools import product
from openai import OpenAI
from tqdm import tqdm
from typing import List, Set
from copy import deepcopy
from utils import pickle_save, pickle_load, make_dir, save_json
from universal_classes import F1Calculator, Oracle
from utils import Retriever
import time

def query_plm(example, 
              template,
              openai_key, 
              model = 'gpt-4-1106-preview', 
              temperature = 0.7,
              max_tokens = 4096, 
              seed = 0,
              ):
    """
    You probably do not need to modify this function. If you want to do multi-turn stuff, either to do NER -> RE or
    to do an extra chatting turn to fix the formatting of the original output, feel free to reuse this function.
    """
    client = OpenAI(api_key=openai_key)
    
    if hasattr(template, 'schema'):
        response = client.chat.completions.create(
            model = model,
            messages = template.make_prompt(example),
            max_tokens = max_tokens,
            functions = [template.schema],
            function_call = {'name': 'extract_relations'},
            seed = seed,
            temperature = temperature
        )
    else:
        response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=template.make_prompt(example),
        max_tokens=max_tokens,
        seed=seed,
        temperature = temperature
    )

    return response

def process_response(response):

    message = response.choices[0].message
    if message.function_call is None:
        # generation = {'relations': {}}
        processed = json.loads(message.content)
    else:
        processed = json.loads(message.function_call.arguments)

    return processed

def load_dataset(dataset_name, split):
    base_directory = 'data/processed'
    dataset = pickle_load(os.path.join(base_directory, dataset_name, f'{split}_data.save'))
    return dataset


# %%
def generate_relations(dataset_name,
                       split,
                       template,
                       openai_key,
                       save_dir,
                       model = 'gpt-4-1106-preview', 
                       temperature = 0.7,
                       max_tokens = 4096, 
                       predicted_relations_filename='predicted_relations.save',
                       max_examples=1,
                       data_seed=0,
                       generate_seed=0
                       ):
    # preliminaries
    make_dir(save_dir)

    random.seed(data_seed)

    # load data
    dataset = load_dataset(dataset_name, split)

    # testing out on a small portion of data
    if max_examples is not None:
        dataset.random_subset(max_examples, data_seed)

    # Get templates for appropriate dataset
    templates = importlib.import_module(f'templates.{dataset_name}')
    template = getattr(templates, template)

    # body
    responses = list()
    generations = list()
    predicted_relations = list()

    for i, el in enumerate(tqdm(dataset, desc='generating')):
        fail_counter = 0
        while fail_counter < 3:
            try:
                response = query_plm(el,
                                     template,
                                     openai_key, 
                                     model, 
                                     temperature,
                                     max_tokens, 
                                     generate_seed + fail_counter * 100)
            except:
                print('Problem with chat completion')
                fail_counter += 1
                response = None
                continue
            
            try:
                generation = process_response(response)
            except:
                print('Problem getting json from response')
                print('response: ', response)
                generation = None
                fail_counter += 1
                continue
            
            try:
                relation_list = generation['relations']
                relations = template.extract_relations(relation_list)
                break
            except:
                print('Problem extracting relations from string')
                print('generation: ', generation)
                relations = None
                fail_counter += 1

        responses.append(response)
        generations.append(generation)
        predicted_relations.append(relations)

        pickle_save(responses, os.path.join(save_dir, 'responses.save'))
        pickle_save(generations, os.path.join(save_dir, 'generations.save'))
        pickle_save(predicted_relations, os.path.join(save_dir, predicted_relations_filename))
        
    return predicted_relations
            
        

def evaluate_performance(dataset_name,
                         split,
                         save_dir,
                         predicted_relations='predicted_relations.save',
                         performance_filename='performance.json',
                         details_filename='details.save',
                         scorer_class='LowercaseScorer',
                         normalized=False,
                         max_examples=1,
                         data_seed=0):
    
    # preliminaries
    make_dir(save_dir)
    random.seed(data_seed)

    # load data
    dataset = load_dataset(dataset_name, split)
    if isinstance(predicted_relations, str):
        predicted_relations = pickle_load(os.path.join(save_dir, predicted_relations))

    # testing out on a small portion of data
    if max_examples is not None:
        dataset.random_subset(max_examples, data_seed)

    # performance classes
    scorer_class = getattr(universal_classes, scorer_class)
    performance_calculator = F1Calculator()

    details_list = list()
    for el_example, el_predicted_relations in tqdm(zip(dataset, predicted_relations), desc='evaluating'):

        # If GPT failed 3 times consecutively (fail_counter < 3), relations should be set() instead of None
        el_example.relations = el_example.relations or set()
        el_predicted_relations = el_predicted_relations or set()
        
        # performance
        if normalized:
            oracle = Oracle(el_example.entities)
            el_predicted_relations = oracle(el_predicted_relations)

        scorer = scorer_class(el_example.relations, el_predicted_relations)
        performance_calculator.update(scorer.TP, scorer.FP, scorer.FN)

        # error analysis details
        details = {'example': el_example,
                   'gold_relations': el_example.relations,
                   'predicted_relations': el_predicted_relations,
                   'performance': deepcopy(scorer),
                   'oracle': oracle if 'oracle' in globals() else None
                   }

        details_list.append(details)

    # calculating overall performance
    performance = performance_calculator.compute()

    # saving results
    save_json(performance, os.path.join(save_dir, performance_filename))
    pickle_save(details_list, os.path.join(save_dir, details_filename))

    return performance

def evaluate_performance_example(el_example,
                         el_predicted_relations,
                         scorer_class='LowercaseScorer',
                         normalized=False,
                         data_seed=0):

    random.seed(data_seed)
    scorer_class = getattr(universal_classes, scorer_class)
    performance_calculator = F1Calculator()

    el_example.relations = el_example.relations or set()
    el_predicted_relations = el_predicted_relations or set()
    
    # performance
    if normalized:
        oracle = Oracle(el_example.entities)
        el_predicted_relations = oracle(el_predicted_relations)

    scorer = scorer_class(el_example.relations, el_predicted_relations)
    performance_calculator.update(scorer.TP, scorer.FP, scorer.FN)

    # error analysis details
    details = {'example': el_example,
                'gold_relations': el_example.relations,
                'predicted_relations': el_predicted_relations,
                'performance': deepcopy(scorer),
                'oracle': oracle if 'oracle' in globals() else None
                }

    # details_list.append(details)

    # calculating overall performance
    performance = performance_calculator.compute()

    return performance

def generate_relations_example(dataset_name,
                       split,
                       template,
                       templateDocs,
                       openai_key,
                       save_dir,
                       model = 'gpt-4-1106-preview', 
                       temperature = 0.7,
                       max_tokens = 4096,
                       max_examples=None,
                       data_seed=1,
                       generate_seed=0,
                       num_docs=5):
    # preliminaries
    save_dir = os.path.join(save_dir, dataset_name + 'Doc')
    save_dir = save_dir + f'_num_docs_{num_docs}'
    if not os.path.exists(save_dir):
        make_dir(save_dir)

    completed = list()
    if os.path.exists(os.path.join(save_dir, f'outfile_{split}.json')):
        curr_progress = list()
        with open(os.path.join(save_dir, f'outfile_{split}.json'), 'r') as file:
            for line in file:
                json_line = json.loads(line.strip())
                curr_progress.append(json_line)

        for d in curr_progress:
            completed.append(d['key'])

    random.seed(data_seed)
    print('Loading dataset...')
    # load data
    dataset = load_dataset(dataset_name, split)
    print('Done.')

    # testing out on a small portion of data
    if max_examples is not None:
        dataset.random_subset(max_examples, data_seed)
    print('Loading templates...')
    # Get templates for appropriate dataset
    templates = importlib.import_module(f'templates.{dataset_name}')
    template = getattr(templates, template)

    # Get templates for appropriate dataset
    templatesDocs = importlib.import_module(f'templates.{dataset_name}')
    templateDocs = getattr(templatesDocs, templateDocs)
    print('Done.')

    # Retriever details
    corpus = 'pubmed'
    db_dir = '../corpus'
    model_name = "ncbi/MedCPT-Query-Encoder"
    print('Building retriever...')
    retriever = Retriever(model_name, corpus, db_dir)
    print(f'Retriever has been built, using checkpoint {model_name}.')
    # body
    # responses = list()
    # responsesDocs = list()
    # generations = list()
    # generationsDocs = list()
    # predicted_relations = list()
    # predicted_relationsDocs = list()
    # performance_zeroshot = list()
    # performance_support_docs = list()

    print('Extract relations without support docs...')

    for i, el in enumerate(tqdm(dataset, desc='generating')):
        if not el.relations:
            print('Current example has no relations, continue to next one...')
            continue
        key = el.title.lower()
        if completed:
            if key in completed:
                print(f'Input with title {key} has been processed, continue to next one...')
                continue
        # print('\n\nExample:\n', el)
        fail_counter = 0
        while fail_counter < 3:
            try:
                response = query_plm(el,
                                     template,
                                     openai_key, 
                                     model, 
                                     temperature,
                                     max_tokens, 
                                     generate_seed + fail_counter * 100,)
            except:
                print('Problem with chat completion')
                fail_counter += 1
                response = None
                continue
            
            try:
                generation = process_response(response)
            except:
                print('Problem getting json from response')
                print('response: ', response)
                generation = None
                fail_counter += 1
                continue
            
            try:
                relation_list = generation['relations']
                relations = template.extract_relations(relation_list)
                break
            except:
                print('Problem extracting relations from string')
                print('generation: ', generation)
                relations = None
                fail_counter += 1

        print('\n\nRelation list extracted without docs:\n', relation_list)
        performance = evaluate_performance_example(el,
                                       relations,
                                       data_seed=data_seed)

        print('\n\nPerformance without docs:\n', performance)

        out_dict = {}
        out_dict['good_docs'] = list()
        out_dict['bad_docs'] = list()
        out_dict['no_difference_docs'] = list()
        out_dict['relation_without_doc'] = relation_list
        out_dict['input_text'] = el.title + el.text
        out_dict['key'] = el.title.lower()
        out_dict['f1_without_doc'] = performance['F1']

        print('Retrieving support docs...')

        
        d, s = retriever.get_relevant_documents(f'Input text:\n\nTitle: {el.title}, Abrastract: {el.text}', k=10)

        #remove same abstract
        docs =[]
        scores = []
        for j in range(len(d)):
            doc = d[j]
            score = s[j]
            if doc['title'] == el.title:
                continue
            docs.append(doc)
            scores.append(score)
        # print('Document retrieved: ', docs)
        # print('Scores retrieved: ', scores)

        out_dict['documents_retrieved'] = [{'id': doc['id'],
                                            'PMID': doc['PMID'],
                                            'title': doc['title'],
                                            'content': doc['content']} for doc in docs[:num_docs]]
        out_dict['document_scores'] = scores[:num_docs]
        print('\n\nExtract relations with support docs...')
        out_dict['relation_with_doc'] = {}

        for m, doc in enumerate(d[:num_docs]):
            time.sleep(5)
            el.docs = [doc]
            # Extract relations with support docs
            fail_counter = 0
            while fail_counter < 3:
                try:
                    responseDocs = query_plm(el,
                                         templateDocs,
                                         openai_key,
                                         model,
                                         temperature,
                                         max_tokens,
                                         generate_seed + fail_counter * 100)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print('Problem with chat completion')
                    fail_counter += 1
                    responseDocs = None
                    continue

                try:
                    generationDocs = process_response(responseDocs)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print('Problem getting json from response')
                    print('response: ', responseDocs)
                    generationDocs = None
                    fail_counter += 1
                    continue

                try:
                    relation_list = generationDocs['relations']
                    relationsDocs = template.extract_relations(relation_list)
                    break
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print('Problem extracting relations from string')
                    print('generation: ', generationDocs)
                    relationsDocs = None
                    fail_counter += 1

            print('\n\nRelation list extracted with docs:\n', relation_list)

            performanceDocs = evaluate_performance_example(el,
                                           relationsDocs,
                                           data_seed=data_seed)

            print(f'\n\nPerformance with doc {(m+1)}/{num_docs}:\n', performanceDocs)

            out_dict['relation_with_doc'][doc['PMID']] = relation_list
            if performanceDocs['F1'] > out_dict['f1_without_doc']:
                out_dict['good_docs'].append(
                    {'id': doc['id'],
                     'PMID': doc['PMID'],
                     'contents': doc['contents'],
                     'F1': performanceDocs['F1'],
                     'difference': performanceDocs['F1'] - out_dict['f1_without_doc']})
            elif performanceDocs['F1'] < out_dict['f1_without_doc']:
                out_dict['bad_docs'].append(
                    {'id': doc['id'],
                     'PMID': doc['PMID'],
                     'contents': doc['contents'],
                     'F1': performanceDocs['F1'],
                     'difference': performanceDocs['F1'] - out_dict['f1_without_doc']})
            else:
                out_dict['no_difference_docs'].append(
                    {'id': doc['id'],
                     'PMID': doc['PMID'],
                     'contents': doc['contents'],
                     'F1': performanceDocs['F1'],
                     'difference': performanceDocs['F1'] - out_dict['f1_without_doc']})

        print(f'Writing results to outfile_{split}.json...')
        with open(os.path.join(save_dir, f'outfile_{split}.json'), 'a') as file:
            json_line = json.dumps(out_dict)
            file.write(json_line + '\n')



def run(dataset_name,
        split,
        template,
        openai_key,
        save_dir,
        model = 'gpt-4-1106-preview', 
        temperature = 0.7,
        max_tokens = 4096, 
        predicted_relations_filename='predicted_relations.save',
        performance_filename='performance.save',
        details_filename='details.save',
        scorer_class='LowercaseScorer',
        normalized=False,
        max_examples=1,
        data_seed=0,
        generate_seed=0
        ):
    
    predicted_relations = generate_relations(dataset_name,
                                             split,
                                             template,
                                             openai_key,
                                             save_dir,
                                             model, 
                                             temperature,
                                             max_tokens, 
                                             predicted_relations_filename,
                                             max_examples,
                                             data_seed,
                                             generate_seed)
    
    # performance = evaluate_performance(dataset_name,
    #                                    split,
    #                                    save_dir,
    #                                    predicted_relations,
    #                                    performance_filename,
    #                                    details_filename,
    #                                    scorer_class,
    #                                    normalized,
    #                                    max_examples,
    #                                    data_seed)
    
    random.seed(data_seed)
    dataset = load_dataset(dataset_name, split)
    dataset.random_subset(max_examples, data_seed)
    
    performance = evaluate_performance_example(dataset,
                                       predicted_relations,
                                       data_seed=data_seed)
    

    return performance

# new function to compare performance with additional documents
def run_biorag(dataset_name,
        split,
        template,
        openai_key,
        save_dir,
        model = 'gpt-4-1106-preview', 
        temperature = 0.7,
        max_tokens = 4096, 
        predicted_relations_filename='predicted_relations.save',
        performance_filename='performance.save',
        details_filename='details.save',
        scorer_class='LowercaseScorer',
        normalized=False,
        max_examples=1,
        data_seed=0,
        generate_seed=0
        ):
    
    random.seed(data_seed)

    # load data
    dataset = load_dataset(dataset_name, split)

    # testing out on a small portion of data
    if max_examples is not None:
        dataset.random_subset(max_examples, data_seed)

    # Get templates for appropriate dataset
    templates = importlib.import_module(f'templates.{dataset_name}')
    template = getattr(templates, template)

    # body
    responses = list()
    generations = list()
    predicted_relations = list()

    for i, el in enumerate(tqdm(dataset, desc='generating')):
        fail_counter = 0
        while fail_counter < 3:
            try:
                response = query_plm(el,
                                     template,
                                     openai_key, 
                                     model, 
                                     temperature,
                                     max_tokens, 
                                     generate_seed + fail_counter * 100)
            except:
                print('Problem with chat completion')
                fail_counter += 1
                response = None
                continue
            
            try:
                generation = process_response(response)
            except:
                print('Problem getting json from response')
                print('response: ', response)
                generation = None
                fail_counter += 1
                continue
            
            try:
                relation_list = generation['relations']
                relations = template.extract_relations(relation_list)
                break
            except:
                print('Problem extracting relations from string')
                print('generation: ', generation)
                relations = None
                fail_counter += 1
    
    predicted_relations = generate_relations(dataset_name,
                                             split,
                                             template,
                                             openai_key,
                                             save_dir,
                                             model, 
                                             temperature,
                                             max_tokens, 
                                             predicted_relations_filename,
                                             max_examples,
                                             data_seed,
                                             generate_seed)
    random.seed(data_seed)
    dataset = load_dataset(dataset_name, split)
    dataset.random_subset(max_examples, data_seed)
    
    performance = evaluate_performance_example(dataset,
                                       predicted_relations,
                                       data_seed=data_seed)
#%% multi-run stuff
def transpose_list(list_of_lists):
    return [[list_of_lists[i][j] for i in range(len(list_of_lists))] for j in range(len(list_of_lists[0]))]

def pool_predictions(list_of_sets):
    return set.union(*list_of_sets)

def majority_vote(list_of_sets):

    unique_relations = set.union(*list_of_sets)
    num_runs = len(list_of_sets)

    majority_relations = set()
    for el_rel in unique_relations:
        presence = 0
        for el_run in list_of_sets:
            if el_rel in el_run:
                presence += 1
        if presence/num_runs >= 0.5:
            majority_relations.add(el_rel)
    return majority_relations

aggregating_functions = {'pool': pool_predictions,
                         'majority': majority_vote}

def aggregate_predictions(predicted_relations_list, aggregate_fun):
    predicted_relations_list = transpose_list(predicted_relations_list)
    return [aggregate_fun(el) for el in predicted_relations_list]
            
def multi_generate_relations(dataset_name,
                       split,
                       templates,
                       aggregate_fun,
                       openai_key,
                       save_dir,
                       model = 'gpt-4-1106-preview', 
                       temperatures = [0.7],
                       max_tokens = 4096, 
                       predicted_relations_filename='predicted_relations.save',
                       max_examples=1,
                       data_seed=0,
                       generate_seeds=[0]):
    
    predicted_relations_list = [generate_relations(dataset_name,
                                                    split,
                                                    el_template,
                                                    openai_key,
                                                    save_dir,
                                                    model, 
                                                    el_temperature,
                                                    max_tokens, 
                                                    predicted_relations_filename,
                                                    max_examples,
                                                    data_seed,
                                                    el_seed) for el_template, el_temperature, el_seed in product(templates, temperatures, generate_seeds)
                                ]
    pickle_save(predicted_relations_list, os.path.join(save_dir, 'multi_predicted_relations.save'))
    predicted_relations = aggregate_predictions(predicted_relations_list, aggregating_functions[aggregate_fun])
    pickle_save(predicted_relations, os.path.join(save_dir, predicted_relations_filename))

    return predicted_relations

def multi_run(dataset_name,
        split,
        templates,
        aggregate_fun,
        openai_key,
        save_dir,
        model = 'gpt-4-1106-preview', 
        temperatures = [0.7],
        max_tokens = 4096, 
        predicted_relations_filename='predicted_relations.save',
        performance_filename='performance.save',
        details_filename='details.save',
        scorer_class='LowercaseScorer',
        normalized=False,
        max_examples=1,
        data_seed=0,
        generate_seeds=[0]
        ):
    
    predicted_relations = multi_generate_relations(dataset_name,
                                                    split,
                                                    templates,
                                                    aggregate_fun,
                                                    openai_key,
                                                    save_dir,
                                                    model, 
                                                    temperatures,
                                                    max_tokens, 
                                                    predicted_relations_filename,
                                                    max_examples,
                                                    data_seed,
                                                    generate_seeds)

    performance = evaluate_performance(dataset_name,
                                       split,
                                       save_dir,
                                       predicted_relations,
                                       performance_filename,
                                       details_filename,
                                       scorer_class,
                                       normalized,
                                       max_examples,
                                       data_seed)

    return performance





# %% running
if __name__ == '__main__':
    fire.Fire()

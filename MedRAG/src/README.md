## Preprocessing

Use preprocessing scripts in `data_processing` to preprocess raw datasets that are in `data/raw`. The preprocessed datasets are in `data/processed`.

## Prompts

Sample `prompt` for CDR dataset:
```
Your task is to extract all chemical-disease relations from a text in which the chemical/drug induces the disease. Note that the chemical or disease names should have appeared in the original input text.
The output should be saved as per the following format:
{'relations':
[
  {
    "chemical": "chemical1",
    "disease": "disease1"
  },
  {
    "chemical": "chemical2",
    "disease": "disease2"
  },
  ...
]
}
The output will be {'relations':[]} if there are no chemical-disease pairs in which the chemical induces the disease expressed in the input text.
With this format, a hypothetical example output for a biomedical text could be the following:
{'relations':
[
  {
    "chemical": "Lidocaine",
    "disease": "cardiac asystole"
  },
  {
    "gene": "daunorubicin",
    "disease": "neutropenia"
  }
]
}
```
## Templates
`Template` should be a class satisfying either OpenAI [function calling](https://platform.openai.com/docs/guides/function-calling) (schema) or [json mode](https://platform.openai.com/docs/guides/text-generation/json-mode) feature.
The schema `Template` should be look like:
```
class Template_schema:
        
    <entity1> = {
            "type": "string",
            "description": "<description>"
        }
    <entity2> = {
            "type": "string",
            "description": "<description>"
        }
    
    items = {
        "type": "object",
        "properties": {
            "<entity1>": ,
            "<entity2>": 
            },
        "description": "<description>"
        }
    
    relations = {
        "type": "array",
        "items": items,
        "required": ["<entity1>", "<entity2>"],
        "description": "<description>"
        }
    
    parameters = {
            "type": "object",
            "properties": {
                "relations": relations
            }
        }

    schema = {
        "name": "extract_relations",
        "description": "Extracts a list of relations from a text.",
        "parameters": parameters,
        "required": ["relations"]
    }

    @classmethod
    def extract_relations(cls, relation_list):
        <YOUR CODE>
        return relations
    
    @classmethod
    def make_prompt(cls, example):
        
        system_content = system_prompt
        
        user_content = f'Title: {example.title}\n\nAbstract: {example.text}'
        
        system = {'role': 'system',
                  'content': system_content}
        
        user = {'role': 'user',
                'content': user_content}
        
        messages = [system, user]

        return messages

```
The json `Template` should be look like:
```
class Template_json:

    @classmethod
    def extract_relations(cls, relation_list):
        <YOUR CODE>
        return relations
    
    @classmethod
    def make_prompt(cls, example):
        
        system_content = system_prompt
        user_content = f'Title: {example.title}\n\nAbstract: {example.text}'
        system = {'role': 'system',
                  'content': "You are a helpful assistant designed to output JSON." + system_content}
        user = {'role': 'user',
                'content': user_content}
        messages = [system, user]

        return messages
```



## Run scripts

```
python main.py run
--model {'gpt-4-1106-preview'}
--dataset_name {ADE | DCE | ChemProt | DDI | CDR | GDA | BioRED}
--split {train | valid | test}
--openai_key {Your API key}
--max_examples 100
--normalized {True | False}
--template {CDRTemplate_json | CDRTemplate_schema | ...}
--save_dir {Directory of output files}
--max_tokens {4096}
--temperatures {0.7}
--data_seed {0}
```

Arguments:

`--model`: GPT models.

`--normalized`: Evaluated on entity level (`True`) or mention level (`False`).

`--template`: `dataset_name` + `Template_json` (inferred models) or `Template_schema` (explicit models). 

`--max_examples`: Maxmium number of examples run by GPT. Set `None` to run on the whole `train/valid/test` dataset.

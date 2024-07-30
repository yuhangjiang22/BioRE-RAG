from typing import Optional, Set, List
from dataclasses import dataclass, field, replace

@dataclass(frozen=True)
class Entity:
    strings: Set[str]
    entity_type: str
    entity_id: Optional[str] = field(default=None)

    def __hash__(self):
        entity_type = self.entity_type
        if self.entity_id:
            # return hash((self.entity_id, entity_type))
            return hash(self.entity_id)
        else:
            string = recursive_lowercase(list(self.strings)[0])
            return hash((string, entity_type))

    def __eq__(self, other):
        is_same_entity_type = self.entity_type == other.entity_type
        is_string_subset = self.strings.issubset(other.strings) or other.strings.issubset(self.strings)
        is_same_id = self.entity_id == other.entity_id and self.entity_id is not None

        return is_same_entity_type and (is_string_subset or is_same_id)

    def add_entity_id(self, _id):
        if isinstance(_id, set):
            _id = list(_id)[0]
        return replace(self, entity_id = _id)

@dataclass(frozen=True)
class Relation:
    entities: Set[Entity]
    relation_type: str

    def __hash__(self):
        return hash((frozenset(self.entities), recursive_lowercase(self.relation_type)))

    def __eq__(self, other):
        return self.relation_type == other.relation_type and self.entities == other.entities

    def add_entity_ids(self, entities_with_ids):
        return replace(self, entities=entities_with_ids)

    def duplicate(self):
        for el_entity in self.entities:
            [deepcopy(self).replace()]

        return [deepcopy(self).replace(my_set={el}) for el in self.my_set]

def recursive_lowercase(data):
    if isinstance(data, str):
        return data.lower()
    elif isinstance(data, list):
        return [recursive_lowercase(el) for el in data]
    elif isinstance(data, set):
        return {recursive_lowercase(el) for el in data}
    elif isinstance(data, dict):
        return {key: recursive_lowercase(value) for key, value in data.items()}
    else:
        return data

def recursive_lowercase2(data):
    if isinstance(data, str):
        return data.lower()
    elif isinstance(data, list):
        return [recursive_lowercase2(el) for el in data]
    elif isinstance(data, set):
        return {recursive_lowercase2(el) for el in data}
    elif isinstance(data, dict):
        return {key: recursive_lowercase2(value) for key, value in data.items()}
    elif isinstance(data, Entity):
        return Entity(recursive_lowercase2(data.strings), recursive_lowercase2(data.entity_type), recursive_lowercase2(data.entity_id))
    elif isinstance(data, Relation):
        return Relation(recursive_lowercase2(data.entities), recursive_lowercase2(data.relation_type))
    else:
        return data 
    
b = {Relation(entities={Entity(strings={'m(3)-machr', 'machr', 'm(3)-muscarinic acetylcholine receptor'}, entity_type='GeneOrGeneProduct', entity_id='24260'), Entity(strings={'aconitine'}, entity_type='ChemicalEntity', entity_id='D000157')}, relation_type='Association'), Relation(entities={Entity(strings={'arrhythmias', 'arrhythmic', 'arrhythmia'}, entity_type='DiseaseOrPhenotypicFeature', entity_id='D001145'), Entity(strings={'ouabain'}, entity_type='ChemicalEntity', entity_id='D010042')}, relation_type='Positive_Correlation'), Relation(entities={Entity(strings={'calcium', 'ca(2+)'}, entity_type='ChemicalEntity', entity_id='D002118'), Entity(strings={'aconitine'}, entity_type='ChemicalEntity', entity_id='D000157')}, relation_type='Positive_Correlation'), Relation(entities={Entity(strings={'m(3)-machr', 'machr', 'm(3)-muscarinic acetylcholine receptor'}, entity_type='GeneOrGeneProduct', entity_id='100379235'), Entity(strings={'aconitine'}, entity_type='ChemicalEntity', entity_id='D000157')}, relation_type='Association'), Relation(entities={Entity(strings={'pilocarpine'}, entity_type='ChemicalEntity', entity_id='D010862'), Entity(strings={'m(3)-machr', 'machr', 'm(3)-muscarinic acetylcholine receptor'}, entity_type='GeneOrGeneProduct', entity_id='24260')}, relation_type='Association'), Relation(entities={Entity(strings={'pilocarpine'}, entity_type='ChemicalEntity', entity_id='D010862'), Entity(strings={'ouabain'}, entity_type='ChemicalEntity', entity_id='D010042')}, relation_type='Negative_Correlation'), Relation(entities={Entity(strings={'calcium', 'ca(2+)'}, entity_type='ChemicalEntity', entity_id='D002118'), Entity(strings={'ouabain'}, entity_type='ChemicalEntity', entity_id='D010042')}, relation_type='Positive_Correlation'), Relation(entities={Entity(strings={'m(3)-machr', 'machr', 'm(3)-muscarinic acetylcholine receptor'}, entity_type='GeneOrGeneProduct', entity_id='100379235'), Entity(strings={'ouabain'}, entity_type='ChemicalEntity', entity_id='D010042')}, relation_type='Association'), Relation(entities={Entity(strings={'arrhythmias', 'arrhythmic', 'arrhythmia'}, entity_type='DiseaseOrPhenotypicFeature', entity_id='D001145'), Entity(strings={'m(3)-machr', 'machr', 'm(3)-muscarinic acetylcholine receptor'}, entity_type='GeneOrGeneProduct', entity_id='24260')}, relation_type='Association'), Relation(entities={Entity(strings={'pilocarpine'}, entity_type='ChemicalEntity', entity_id='D010862'), Entity(strings={'arrhythmias', 'arrhythmic', 'arrhythmia'}, entity_type='DiseaseOrPhenotypicFeature', entity_id='D001145')}, relation_type='Negative_Correlation'), Relation(entities={Entity(strings={'pilocarpine'}, entity_type='ChemicalEntity', entity_id='D010862'), Entity(strings={'aconitine'}, entity_type='ChemicalEntity', entity_id='D000157')}, relation_type='Negative_Correlation'), Relation(entities={Entity(strings={'pilocarpine'}, entity_type='ChemicalEntity', entity_id='D010862'), Entity(strings={'ventricular tachycardia and fibrillation'}, entity_type='DiseaseOrPhenotypicFeature', entity_id='D017180|D014693')}, relation_type='Negative_Correlation'), Relation(entities={Entity(strings={'arrhythmias', 'arrhythmic', 'arrhythmia'}, entity_type='DiseaseOrPhenotypicFeature', entity_id='D001145'), Entity(strings={'aconitine'}, entity_type='ChemicalEntity', entity_id='D000157')}, relation_type='Positive_Correlation'), Relation(entities={Entity(strings={'4-damp', '4-diphenylacetoxy-n-methylpiperidine-methiodide'}, entity_type='ChemicalEntity', entity_id='C042375'), Entity(strings={'pilocarpine'}, entity_type='ChemicalEntity', entity_id='D010862')}, relation_type='Negative_Correlation'), Relation(entities={Entity(strings={'calcium', 'ca(2+)'}, entity_type='ChemicalEntity', entity_id='D002118'), Entity(strings={'pilocarpine'}, entity_type='ChemicalEntity', entity_id='D010862')}, relation_type='Negative_Correlation'), Relation(entities={Entity(strings={'m(3)-machr', 'machr', 'm(3)-muscarinic acetylcholine receptor'}, entity_type='GeneOrGeneProduct', entity_id='100379235'), Entity(strings={'arrhythmias', 'arrhythmic', 'arrhythmia'}, entity_type='DiseaseOrPhenotypicFeature', entity_id='D001145')}, relation_type='Association'), Relation(entities={Entity(strings={'m(3)-machr', 'machr', 'm(3)-muscarinic acetylcholine receptor'}, entity_type='GeneOrGeneProduct', entity_id='100379235'), Entity(strings={'pilocarpine'}, entity_type='ChemicalEntity', entity_id='D010862')}, relation_type='Association'), Relation(entities={Entity(strings={'ouabain'}, entity_type='ChemicalEntity', entity_id='D010042'), Entity(strings={'m(3)-machr', 'machr', 'm(3)-muscarinic acetylcholine receptor'}, entity_type='GeneOrGeneProduct', entity_id='24260')}, relation_type='Association'), Relation(entities={Entity(strings={'m(3)-machr', 'machr', 'm(3)-muscarinic acetylcholine receptor'}, entity_type='GeneOrGeneProduct', entity_id='100379235'), Entity(strings={'4-damp', '4-diphenylacetoxy-n-methylpiperidine-methiodide'}, entity_type='ChemicalEntity', entity_id='C042375')}, relation_type='Negative_Correlation'), Relation(entities={Entity(strings={'4-damp', '4-diphenylacetoxy-n-methylpiperidine-methiodide'}, entity_type='ChemicalEntity', entity_id='C042375'), Entity(strings={'m(3)-machr', 'machr', 'm(3)-muscarinic acetylcholine receptor'}, entity_type='GeneOrGeneProduct', entity_id='24260')}, relation_type='Negative_Correlation')}

c = recursive_lowercase2(b)

print(c)
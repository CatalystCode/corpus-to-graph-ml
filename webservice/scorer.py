import datetime
import itertools
import os
import pickle
import sys
import sklearn
import urllib
from mirna_detector import is_mirna
from os import path

# add the dir above to path
# TODO: make this more standard...
current_dir_path = path.dirname(path.realpath(__file__))
sys.path.append(path.join(path.dirname(current_dir_path),"packages"))
from model_tools import ScoringModel

# TODO: Create dev and prod envs..
model_file_name = r"scoring_model.pkl"
model_directory_path = path.join(current_dir_path, r"model")
model_file_path = path.join(model_directory_path,model_file_name)

scoring_model = None

try:
    scoring_model = ScoringModel.from_file(model_file_path)
except Exception as e:
    print "Failed loading model: %s"%e

def get_text_from_entity_dict(e):
    if e.has_key("origin"):
        return e["origin"]
    if e.has_key("value"):
        return e["value"]
    return None

def get_version():
    if (scoring_model == None):
        raise Exception("No model file loaded, or bad model exists")
    return scoring_model.version

def get_temp_model_path():
    return path.join(model_directory_path, model_file_name + "_" + datetime.datetime.now().strftime("%y%m%d_%H%M%S"))

def load_model_from_url(url):
    # TODO: move this into a class..
    global scoring_model
    url_opener = urllib.URLopener()
    temp_model_path =  get_temp_model_path()
    url_opener.retrieve(url, temp_model_path)

    # try to load the model:
    try:
        temp_model = ScoringModel.from_file(temp_model_path)
    except Exception as e:
        print "Failed to load donwloaded model: %s"%e
        os.remove(temp_model_path)
        raise RuntimeError("Failed to load donwloaded model! error: %s"%e)

    # update model:
    scoring_model = temp_model

    # delete existing model
    if (path.isfile(model_file_path)):
        os.remove(model_file_path)
    os.rename(temp_model_path, model_file_path)


# TODO: move this to an object with an init function...
def evaluate_score(sentence, entities):
    if (scoring_model == None):
        raise Exception("No model file loaded, or bad model exists")

    for e in entities:
        if e.has_key("type"):
            e["type"] = e["type"].lower()
    
    # TODO:
    # We merge the entities here such that there are no overlaps, and also check for the mirna specifically
    # this should basically solved on the entity recognition side
    # we should consider removing this part when this issue is solved
    filtered_entities = []
    for entity in entities:
        # check if we don't have collisions, always take the longer string
        objects_to_remove = []
        add_entity = True

        start_index = int(entity["from"])
        end_index = int(entity["to"])
        text = get_text_from_entity_dict(entity)

        for e in filtered_entities:
            if (start_index <= e["start_index"] and end_index >= e["end_index"] 
                or start_index <= e["start_index"] and end_index >= e["start_index"]
                or start_index <= e["end_index"] and end_index >= e["end_index"]):
                current_len = len(text)
                second_len = len(e["text"])
                if (current_len <= second_len):
                    add_entity = False
                    break
                else:
                    objects_to_remove.append(e)

        for e in objects_to_remove:
            filtered_entities.remove(e)

        # todo: for now using the mirna detector to detect miRNA since it seems that
        # the current results are currently not accurate
        if (not add_entity):
            continue

        type = entity["type"]
        if (is_mirna(text)):
            type = "mirna"
        entity["type"] = type

        filtered_entities.append({
                "text" : text,
                "type" : type,
                "start_index" : start_index,
                "end_index" : end_index,
                "original_entity" : entity
            })

    # for now just return the same result for all pairs of gens / miRNA entities:
    mirna_entities = [e for e in filtered_entities if (e["type"]=="mirna")]
    gene_entities = [e for e in filtered_entities if (e["type"]=="gene")]
    scores = []
    
    for p in itertools.product(mirna_entities, gene_entities):
        context = {"pair_entities" :[
                p[0],
                p[1]
            ],
            "all_entities" : filtered_entities
        }

        score = scoring_model.score(sentence, context)
        scores.append((score, (p[0]["original_entity"],p[1]["original_entity"])))
    
    return scores

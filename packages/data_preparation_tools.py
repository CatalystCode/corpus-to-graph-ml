import math
import numpy as np
import nltk
import urllib
import requests
import re
from mirna_detector import is_mirna
from os import path
from sklearn.cross_validation import train_test_split


#constants:
DEFAULT_VARIABLE_FORMAT_TABLE = {'gene': 'GGVARENTTY%dGG',
                                'mirna' : 'MMVARENTTY%dMM' }

PAIR_ENTITTY_VARIABLE_NAMES = {'gene': 'GGVARENTTYGG',
                                'mirna' : 'MMVARENTTYMM' }

OTHER_ENTITY_VARIABLE_TABLE = {'gene': 'GGVARENTTYOTRGG',
                                'mirna' : 'MMVARENTTYOTRMM' }

# expects output in GNAT's output format
ENTITY_RECOGNITION_SERVICE_URL_FORMAT = ""

DEFAULT_SENTENCE_COLUMN = 8
DEFAULT_LABEL_COLUMNS = [4,5]
DEFAULT_MIRNA_ENTITY_COLUMN = 10
DEFAULT_GENE_ENTITY_COLUMN = 11
DEFAULT_MIN_SENTENCE_LENGTH = 6
DEFAULT_EXTRA_WORDS_COUNT = 3
ENTITY_REGEX = re.compile(r"GGVARENTTYGG|MMVARENTTYMM")
DEFAULT_TEST_SIZE = 0.25

# TODO: we should probably take this from a file
DEFAULT_NON_ENTITY_DICTIONARY = ['and', 'lab']
DEFAULT_MIN_ENTITY_LENGTH = 3

# Object to pass along and contains the input to the different transformations
class InputData:
    def __init__(self, data_train, data_test, contexts_train, contexts_test):
        self.data_train = data_train
        self.data_test = data_test
        self.contexts_train = contexts_train
        self.contexts_test = contexts_test

# entities is a list of dictionaries in the format of {"text" : "...", type:
# "...", }
def replace_entities_with_variables(text, entities, variable_format_table=OTHER_ENTITY_VARIABLE_TABLE):
    text_parts = []
    
    for entity in entities:
        entity_replacement = variable_format_table[entity["type"]]
        text = text.replace(entity["text"], entity_replacement)
        
    return text

# entities is a list of dictionaries in the format of {"text" : "...", type:
# "...", }
def replace_entities_with_variables_old(text, entities, variable_name_table=DEFAULT_VARIABLE_FORMAT_TABLE):
    text_parts = []
    
    locations_by_type = {}
    
    for entity in entities:
        index = 0
        index = text.find(entity["text"])
        if (index == -1):
            continue
        
        if (not locations_by_type.has_key(entity["type"])):
            locations_by_type[entity["type"]] = {}
            
        locations_by_type[entity["type"]][index] = entity
    
    for t in locations_by_type:
        entities_to_variables = {}
        index = 1
        # assign a variable to each entity
        for i in sorted(locations_by_type[t]):
            entities_to_variables[locations_by_type[t][i]["text"]] = variable_format_table[t] % index
            index = index + 1

        for entity in entities_to_variables:
            text = text.replace(entity, entities_to_variables[entity])
        
    return text

texts_entities_dictionary = {}
def get_entities_for_text(text):
    # in case we already have the result...
    if (texts_entities_dictionary.has_key(text)):
        return texts_entities_dictionary[text]
    
    url = ENTITY_RECOGNITION_SERVICE_URL_FORMAT % (urllib.quote(text))
    r = requests.get(url)
    if (r.status_code != 200):
        # TODO: Throw exception...
        print "got bad error code:%d" % r.status_code
        return None
    
    parsed_entities = []
    entities = []
    for line in r.text.split("\n"):
        if (len(line) == 0):
            continue
        line_parts = line.split("\t")
        entity_type = line_parts[2]
        start_index = int(line_parts[5])
        end_index = int(line_parts[6])
        text = line_parts[7]
        
        # only look at genes for now
        if (entity_type != 'gene' and entity_type != 'mirna'):
            continue
        
        # check if we don't have collisions, always take the longer string
        objects_to_remove = []
        add_entity = True
        for e in parsed_entities:
            if (start_index <= e["startIndex"] and end_index >= e["endIndex"] or start_index <= e["startIndex"] and end_index >= e["startIndex"] or start_index <= e["endIndex"] and end_index >= e["endIndex"]):
                currentLen = len(text)
                second_len = len(e["text"])
                if (currentLen <= second_len):
                    add_entity = False
                    break
                else:
                    objects_to_remove.append(e)

        for e in objects_to_remove:
            parsed_entities.remove(e)

        if (add_entity):
            parsed_entities.append({
                    "text" : text,
                    "startIndex" : start_index,
                    "endIndex" : end_index
                })

    for e in parsed_entities:
        if (not e["text"] in entities):
            entities.append(e["text"])

    # merge entities in case there is overlap

    return entities

# utility function to add text to entities data to the dictionary
def import_to_texts_entities_dictonary_from_file(file_path):
    with open(file_path) as handle:
        for line in handle:
            parts = line.rstrip().split("\t")
            texts_entities_dictionary[parts[0]] = parts[1:]
            

# the results of the entityt recognition might be noisy
def filter_entities(entities, non_entities_dictionary=DEFAULT_NON_ENTITY_DICTIONARY, min_entity_length=DEFAULT_MIN_ENTITY_LENGTH):
    filtered_entities = []
    for e in entities:
        if (len(e) < min_entity_length):
            continue
        if (e in non_entities_dictionary):
            continue
        
        filtered_entities.append(e)
        
    return filtered_entities

def entity_list_to_descriptors(entities):
    result = []
    for e in entities:
        type = "mirna" if is_mirna(e) else "gene"
        result.append({"text" : e, "type" : type})
    return result

def extract_and_replace_entities(text, context=None, return_descriptors=False):
    if (context != None and context.has_key("pair_entities")):
        replaced_text = replace_entities_with_variables(text, context["pair_entities"], PAIR_ENTITTY_VARIABLE_NAMES)
    if (context != None and context.has_key("all_entities")):
        # TODO: run unification logic here as well?
        entity_descriptors = context["all_entities"]
    else:
        entities = get_entities_for_text(text)
        entities = filter_entities(entities)
        entity_descriptors = entity_list_to_descriptors(entities)
    
    replaced_text = replace_entities_with_variables(replaced_text, entity_descriptors)
    if (return_descriptors):
        return (replaced_text, descriptors)
    else:
        return replaced_text

# extract data from CSV/TSV file
def extract_sentences(input_file_path,
                     sentence_columns=DEFAULT_SENTENCE_COLUMN, mirna_entity_column=DEFAULT_MIRNA_ENTITY_COLUMN,
                     gene_entity_column=DEFAULT_GENE_ENTITY_COLUMN, label_column_indices=None, 
                     label_tag=None, sample_size=-1):
    sentences = []
    labels = []
    contexts = []
    all_sentences = {}
    with open(input_file_path) as input:
        for line in input:
            splitted_line = line.rstrip().split("\t")
            sentence = splitted_line[sentence_columns]
            
            # TODO: randomly sample one sentence instead of just the first one in the list
            # We do this in order to make sure that there isn't a bias in the model towards sentences that
            # that appear more than other ones
            if (all_sentences.has_key(sentence)):
                continue

            all_sentences[sentence] = 1
            
            contexts.append({"pair_entities" : [{"text" : splitted_line[mirna_entity_column], "type": "mirna"},
                        {"text" : splitted_line[gene_entity_column], "type": "gene"}]})
                
            if (label_column_indices != None):
                label = splitted_line[label_column_indices[0]]
                for i in label_column_indices[1:]:
                    label = label + "_" + splitted_line[i]
                labels.append(label)
            
            sentences.append(sentence)
            
    if (sample_size != -1):
        indices = np.random.choice(len(sentences), sample_size)
        sentences = [sentences[index] for index in indices]
        contexts = [contexts[index] for index in indices]
    
    if (label_tag != None):
        labels = [label_tag] * len(sentences)
    return (sentences, labels, contexts)
    
def extract_sentences_with_multiclass_labels(input_file_path, sample_size=-1):
    return extract_sentences(input_file_path, label_column_indices=DEFAULT_LABEL_COLUMNS, sample_size=sample_size)

def is_sequence(arg):
    return (not hasattr(arg, "strip") and hasattr(arg, "__getitem__") or hasattr(arg, "__iter__"))

def write_lines_or_tuples_to_file(lines, output_file_path, seperator="\t"):
    are_tuples = is_sequence(lines[0])
    with open(output_file_path, "w") as handle:
        for o in sentences:
            if are_tuples:
                text = o[0]
                for item in o[1:]:
                    text = text + seperator + item
            else:
                text = o
            handle.write(text + "\n")

# get all entities for a list of sentences:
def get_entities_for_file(in_file_path, out_file_path):
    sentences = extract_sentences(in_file_path)
    with open(out_file_path) as out_handle:
        for s,i in zip(sentences, xrange(len(sentences))):
            print "%d of %d" % (i, len(sentences))
            entities = get_entities_for_text(s)
            out_handle.write(sentence + "\t" + "\t".join(entities) + "\n")

def trim_sentence_around_entities(text , context=None, min_length=DEFAULT_MIN_SENTENCE_LENGTH, extra_words_count=DEFAULT_EXTRA_WORDS_COUNT):
    sentence_parts = text.split()
    
    if (len(sentence_parts) < min_length):
        return text
    
    first_index = -1
    last_index = -1
    
    for part,i in zip(sentence_parts, xrange(len(sentence_parts))):
        if (ENTITY_REGEX.match(part)):
            if (first_index == -1):
                first_index = i
            last_index = i
    
    size = last_index - first_index + extra_words_count * 2
    
    # ensure
    if (size < min_length):
        extra_words_count = extra_words_count + math.ceil((min_length - size) / 2)
    
    first_index = max(0, first_index - extra_words_count)
    last_index = min(len(sentence_parts), last_index + extra_words_count + 1)
    
    trimmed_sentence_parts = sentence_parts[first_index:last_index]
    return " ".join(trimmed_sentence_parts)

# langueage based sentence process
def normalize_text(sent, context=None):
    return sent.lower()

# stop words removal
def remove_stop_words(sent, context=None):
    processed_tokens = []
    tokens = nltk.word_tokenize(sent)
    for t in tokens:
        # ignore stop words
        if (t in nltk.corpus.stopwords.words('english') or len(t) < 2):
            continue
        processed_tokens.append(t)

    return " ".join(processed_tokens)

# digits removal
def remove_all_digit_tokens(sent, context=None):
    processed_tokens = []
    tokens = nltk.word_tokenize(sent)
    for t in tokens:
        # ignore stop words
        if (t.isdigit()):
            continue
        processed_tokens.append(t)

    return " ".join(processed_tokens)

# run stemmer on the words
def stem_text(sent, context=None):
    processed_tokens = []
    tokens = nltk.word_tokenize(sent)
    porter = nltk.PorterStemmer()
    for t in tokens:
        t = porter.stem(t)
        processed_tokens.append(t)

    return " ".join(processed_tokens)

# Split to train and test sample sets:
def split_to_test_and_train(data, labels, entities, test_size=DEFAULT_TEST_SIZE):
    d_train, d_test, l_train, l_test, c_train, c_test = train_test_split(data, labels, entities, test_size=test_size)
    d_test_2 = []
    l_test_2 = []
    c_test_2 = []

    train_dict = {}
    for d in d_train:
        train_dict[d] = 1

    for d,l,c in zip(d_test, l_test, c_test):
        if (train_dict.has_key(d)):
            continue
        d_test_2.append(d)
        l_test_2.append(l)
        c_test_2.append(c)

    return (d_train, d_test_2, l_train, l_test_2, c_train, c_test_2)

# utility to extracts entities from preproceseed files
def extract_entities_from_entity_file(input_file_paths, out_file_path):
    all_entities = {}

    for file_path in input_file_paths:
        with open(file_path) as handle:
            for l in handle:
                parts = l.rstrip().split("\t")
                for e in parts[1:]:
                    all_entities[e] = 1

    with open(out_file_path, "w") as handle:
        for e in sorted(all_entities.keys()):
            handle.write(e + "\n")

def run_step(step_name, step_func, inputs_dict, required):
    temp_dict = {}
    for k in inputs_dict:
        
        if (required != None):
            found_all = True
            for r in required:
                if (k.find(r) == -1):
                    found_all = False
            if (not found_all):
                continue
        
        result_train = []
        result_test = []
        
        for l,c in zip(inputs_dict[k].data_train, inputs_dict[k].contexts_train):
            result_train.append(step_func(l, context=c))
            
        for l,c in zip(inputs_dict[k].data_test, inputs_dict[k].contexts_test):
            result_test.append(step_func(l, context=c))
            
        temp_dict[k + "_" + step_name] = InputData(result_train, result_test, 
                                                   inputs_dict[k].contexts_train, inputs_dict[k].contexts_test)
    
    for k in temp_dict:
        inputs_dict[k] = temp_dict[k]

def run_step_unlabeled_data(step_name, step_func, inputs_dict, required):
    temp_dict = {}
    for k in inputs_dict:
        
        if (required != None):
            found_all = True
            for r in required:
                if (k.find(r) == -1):
                    found_all = False
            if (not found_all):
                continue
        
        results = []

        for l in inputs_dict[k]:
            results.append(step_func(l))
            
        temp_dict[k + "_" + step_name] = results
    
    for k in temp_dict:
        inputs_dict[k] = temp_dict[k]

# 3rd part specify required step
TRANSFORMATION_STEPS = [('entities', extract_and_replace_entities),
            ('trim', trim_sentence_around_entities, ['entities']),
            ('normalize', normalize_text, ['entities']),
            ('rmdigits', remove_all_digit_tokens, ['entities']),
#            ('rmstopwords', remove_stop_words, ['entities'])
#            ('stem', stem_text, ['entities'])
            ]

TRANSFORMATION_STEPS_UNLABELED = [('entities', extract_and_replace_entities),
            ('normalize', normalize_text, ['entities']),
            ('rmdigits', remove_all_digit_tokens, ['entities']),
            ('rmstopwords', remove_stop_words, ['entities'])]

# steps is an array of tuples, with the first as name and the second is the processing function
def run_transformations_on_single_sentence(text, context, steps=TRANSFORMATION_STEPS):
    current = text
    for s in steps:
        current = s[1](current, context=context)
    return current

def run_transformations_on_data(sentences, labels, contexts, output_files_prefix, output_dir, write_context_to_files=True):
    # split to train and test:
    print "before data split"
    s_train, s_test, l_train, l_test, c_train, c_test = split_to_test_and_train(sentences, labels, contexts)
    inputs = {
        'data' : InputData(s_train, s_test, c_train, c_test)
    }

    print "splitted data"
    # run each step on all of the already existing ones
    # TODO: pass entities metadata and let the trimming work even without the
    # regexes..
    
    for s in TRANSFORMATION_STEPS:
        print "running step:%s" % (s[0])
        required = None
        if (len(s) > 2):
            required = s[2]
        run_step(s[0], s[1], inputs, required)
        
    # todo: write outputs to files
    for name in inputs:
        train_file_path = path.join(output_dir, output_files_prefix + "_" + name + "_train.tsv")
        test_file_path = path.join(output_dir, output_files_prefix + "_" + name + "_test.tsv")
        
        train_data = inputs[name].data_train
        test_data = inputs[name].data_test
        
        with open(train_file_path, "w") as handle:
            for text,label in zip(train_data,l_train):
                if (len(text.strip()) == 0):
                    continue
                handle.write("%s\t%s\n" % (label, text))
                
        with open(test_file_path, "w") as handle:
            for text,label in zip(test_data,l_test):
                if (len(text.strip()) == 0):
                    continue
                handle.write("%s\t%s\n" % (label, text))

    if (write_context_to_files):
        train_context_file_path = path.join(output_dir, output_files_prefix + "_context_train.tsv")
        test_context_file_path = path.join(output_dir, output_files_prefix + "_context_test.tsv")

        #({"pair_entities" : [{"text" : splitted_line[mirna_entity_column], "type": "mirna"},
        #                {"text" : splitted_line[gene_entity_column], "type": "gene"}]})

        with open(train_context_file_path, "w") as handle:
            for c in c_train:
                # TODO: generalize this, for now we just do this quick and dirty..
                entity_1_type = c["pair_entities"][0]["type"]

                if (entity_1_type == "mirna"):
                    mirna = c["pair_entities"][0]["text"]
                    gene = c["pair_entities"][1]["text"]
                else:
                    mirna = c["pair_entities"][1]["text"]
                    gene = c["pair_entities"][0]["text"]

                handle.write(mirna + "\t" + gene + "\n")

        with open(test_context_file_path, "w") as handle:
            for c in c_test:
                # TODO: generalize this, for now we just do this quick and dirty..
                entity_1_type = c["pair_entities"][0]["type"]

                if (entity_1_type == "mirna"):
                    mirna = c["pair_entities"][0]["text"]
                    gene = c["pair_entities"][1]["text"]
                else:
                    mirna = c["pair_entities"][1]["text"]
                    gene = c["pair_entities"][0]["text"]

                handle.write(mirna + "\t" + gene + "\n")
        


def write_lines_to_file(lines, out_file_path):
    with open(out_file_path,"w") as handle:
        for line in lines:
            handle.write(line + "\n")

def read_lines_from_file(in_file_path):
    with open(in_file_path) as handle:
        lines = [line.rstrip() for line in handle]
    return lines

def sample_from_file(in_file_path, out_file_path, sample_size):
    lines = read_lines_from_file(in_file_path)
    sampled = [lines[index] for index in np.random.choice(len(lines), sample_size)]
    write_lines_to_file(sampled, out_file_path)

def run_data_preparation_pipeline(positive_samples_file_path, negative_samples_file_path, 
                                  output_files_prefix, output_dir, run_multiclass=False):
    # two classes case:
    pos_sentences, pos_labels, pos_contexts = extract_sentences(positive_samples_file_path, label_tag='RELATION')
    neg_sentences, neg_labels, neg_contexts = extract_sentences(negative_samples_file_path, label_tag='NO_RELATION', sample_size = len(pos_sentences))
    contexts = pos_contexts + neg_contexts
    sentences = pos_sentences + neg_sentences
    labels = pos_labels + neg_labels
    
    print "producing inputs for binary case..."

    run_transformations_on_data(sentences, labels, contexts, output_files_prefix + "_binary", output_dir)
    
    if (run_multiclass):
        # multi class case:
        print "producing inputs for multiclass case..."
        pos_sentences, pos_labels, pos_entities = extract_sentences_with_multiclass_labels(positive_samples_file_path)
        sentences = pos_sentences + neg_sentences
        labels = pos_labels + neg_labels
        contexts = pos_contexts + neg_contexts
    
        run_transformations_on_data(sentences, labels, contexts, output_files_prefix + "_multiclass", output_dir)


def run_transformations_on_unlabeled_data(sentences, output_files_prefix, output_dir):
    inputs = {
        'data' : sentences
    }
    
    # run each step on all of the already existing ones
    # TODO: pass entities metadata and let the trimming work even without the
    # regexes..
    
    for s in TRANSFORMATION_STEPS_UNLABELED:
        print "running step:%s" % (s[0])
        required = None
        if (len(s) > 2):
            required = s[2]
        run_step_unlabeled_data(s[0], s[1], inputs, required)
        
    # todo: write outputs to files
    for name in inputs:
        out_file_path = path.join(output_dir, output_files_prefix + "_" + name + ".txt")
        
        with open(out_file_path, "w") as handle:
            for text in inputs[name]:
                if (len(text.strip()) == 0):
                    continue
                handle.write("%s\n" % (text))


def run_unlabeled_data_preparation_pipeline(samples_file_path, output_files_prefix, output_dir):
    sentences = read_lines_from_file(samples_file_path)
    run_transformations_on_unlabeled_data(sentences, output_files_prefix + "_binary", output_dir)
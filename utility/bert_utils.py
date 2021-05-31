import os
from utility.sequence_classification import MySequenceClassification
from transformers import AutoTokenizer
import torch


def get_tokenizer(model_name):
    # bert_model_name = bert_base / scibert / biobert
    tokenizer = None
    if model_name == 'bert_base':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    elif model_name == 'scibert':
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased", do_lower_case=False)
    elif model_name == 'biobert':
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1", do_lower_case=False)
    else:
        raise Exception('Tokenizer should be one of the following; bert-base / scibert / biobert')
    
    return tokenizer

def get_model(model_name, num_classes):
    # num_classes differ whether the label is predicate or framenet
    model = None
    if model_name == 'bert_base':
        model = MySequenceClassification.from_pretrained("bert-base-cased", num_labels=num_classes)
    elif model_name == 'scibert':
        model = MySequenceClassification.from_pretrained("allenai/scibert_scivocab_cased", num_labels=num_classes)
    elif model_name == 'biobert':
        model = MySequenceClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1", num_labels=num_classes)
    else:
        raise Exception('BERT model should be one of the following; bert-base / scibert / biobert')

    return model

def load_model(model, model_file_name):
    # Used for prediction
    state = torch.load(os.path.join('./models_BERT/'+model_file_name), map_location='cpu')
    model.load_state_dict(state['model'])
    return model
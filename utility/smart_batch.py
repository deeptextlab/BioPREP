from utility.helper_utils import good_update_interval

import random
import torch

def tokenize_truncate(text_samples, labels, tokenizer, max_len):

    full_input_ids = []

    if labels is None:  # pred
        # Tokenize all training examples
        print('Tokenizing {:,} samples...'.format(len(text_samples)))
        update_interval = good_update_interval(total_iters=len(text_samples), num_desired_updates=10)
    else:  # train
        # Tokenize all training examples
        print('Tokenizing {:,} samples...'.format(len(labels)))  
        update_interval = good_update_interval(total_iters=len(labels), num_desired_updates=10)
        
    # For all sentences
    for text in text_samples:  
        if ((len(full_input_ids) % update_interval) == 0):
            print('  Tokenized {:,} samples.'.format(len(full_input_ids)))

        # Tokenization without padding
        input_ids = tokenizer.encode(text=text,              # Text to encode.
                                    add_special_tokens=True, # Do add specials.
                                    max_length=max_len,      # Do Truncate!
                                    truncation=True,         # Do Truncate!
                                    padding=False)           # DO NOT pad.
                                    
        # Append input (encoded with token id) to full_input_ids
        full_input_ids.append(input_ids)
        
    print('DONE.')
    print('{:>10,} samples\n'.format(len(full_input_ids)))

    return full_input_ids


def select_batches(full_input_ids, labels, batch_size):
    if labels is None:  # pred
        # Keeping the test data order
        samples = list(full_input_ids)
        print('{:>10,} samples without sorting for prediction\n'.format(len(samples)))

        # List to add each batch
        batch_ordered_sentences = []
        print('Creating batches of size {:}...'.format(batch_size))
        
        update_interval = good_update_interval(total_iters=len(samples), num_desired_updates=10)

        # Until all the samples are batched..
        while len(samples) > 0:
            if ((len(batch_ordered_sentences) % update_interval) == 0 \
                and not len(batch_ordered_sentences) == 0):
                print('  Selected {:,} batches.'.format(len(batch_ordered_sentences)))

            to_take = min(batch_size, len(samples))

            # Select index in order
            select = 0

            # Batch
            batch = samples[select:(select + to_take)]

            # Batch token
            batch_ordered_sentences.append([s for s in batch])

            # Remove batch from sample
            del samples[select:select + to_take]

        print('\n  DONE - Selected {:,} batches.\n'.format(len(batch_ordered_sentences)))
        batch_ordered_labels = None
        
    else:  # train
        # Order by token length
        samples = sorted(zip(full_input_ids, labels), key=lambda x: len(x[0]))
        print('{:>10,} samples after sorting\n'.format(len(samples)))

        # List for each batch
        batch_ordered_sentences = []
        batch_ordered_labels = []
        print('Creating batches of size {:}...'.format(batch_size))

        update_interval = good_update_interval(total_iters=len(samples), num_desired_updates=10)
        
        # Until all samples are batched..
        while len(samples) > 0:
            if ((len(batch_ordered_sentences) % update_interval) == 0 \
                and not len(batch_ordered_sentences) == 0):
                print('  Selected {:,} batches.'.format(len(batch_ordered_sentences)))

            to_take = min(batch_size, len(samples))

            # Select random index
            select = random.randint(0, len(samples) - to_take)

            # Batch
            batch = samples[select:(select + to_take)]
            
            # Batch token
            batch_ordered_sentences.append([s[0] for s in batch])
            # Batch label
            batch_ordered_labels.append([s[1] for s in batch])

            # Remove batch from sample
            del samples[select:select + to_take]

        print('\n  DONE - Selected {:,} batches.\n'.format(len(batch_ordered_sentences)))
        
    return batch_ordered_sentences, batch_ordered_labels


def add_padding(tokenizer, batch_ordered_sentences, batch_ordered_labels=None):
    print('Padding out sequences within each batch...')

    if batch_ordered_labels!=None:  # train
        py_inputs = []
        py_attn_masks = []
        py_labels = []

        # (Similar token length) Create padded input to each batch
        for (batch_inputs, batch_labels) in zip(batch_ordered_sentences, batch_ordered_labels):
            batch_padded_inputs, batch_attn_masks = batch_result(tokenizer, batch_inputs)

            # Save each batch input result
            py_inputs.append(torch.tensor(batch_padded_inputs))
            py_attn_masks.append(torch.tensor(batch_attn_masks))
            py_labels.append(torch.tensor(batch_labels))
        
        print('  DONE.')
    
    else:  # pred
        py_inputs = []
        py_attn_masks = []

        # (Similar token length) Create padded input to each batch
        for batch_inputs in batch_ordered_sentences:
            batch_padded_inputs, batch_attn_masks = batch_result(tokenizer, batch_inputs)

            # Save each batch input result
            py_inputs.append(torch.tensor(batch_padded_inputs))
            py_attn_masks.append(torch.tensor(batch_attn_masks))
        
        py_labels = None
        print('  DONE.')
    
    # Model's final input (Final input of model)
    return py_inputs, py_attn_masks, py_labels

def batch_result(tokenizer, batch_inputs):
    batch_padded_inputs = []
    batch_attn_masks = []
    
    # Longest sentence in batch
    max_size = max([len(sen) for sen in batch_inputs])

    # About each sentence
    for sen in batch_inputs:
        
        # Padding to add
        num_pads = max_size - len(sen)

        # Add padding
        padded_input = sen + [tokenizer.pad_token_id]*num_pads

        # Attention mask
        attn_mask = [1] * len(sen) + [0] * num_pads

        # Result of each batch
        batch_padded_inputs.append(padded_input)
        batch_attn_masks.append(attn_mask)
    
    return batch_padded_inputs, batch_attn_masks


def make_smart_batches(tokenizer=None, max_len=None, text_samples=None, labels=None, batch_size=None):
    '''
    Tokenize a sentence without padding - Create batches of sentences of similar tokenized length - Create an input with padding added
    When parameter labels==None, this function is used for prediction.
    '''
    print('Creating Smart Batches from {:,} examples with batch size {:,}...\n'.format(len(text_samples), batch_size))

    full_input_ids = tokenize_truncate(text_samples, labels, tokenizer, max_len)
    batch_ordered_sentences, batch_ordered_labels = select_batches(full_input_ids, labels, batch_size)
    py_inputs, py_attn_masks, py_labels = add_padding(tokenizer, batch_ordered_sentences, batch_ordered_labels)

    return py_inputs, py_attn_masks, py_labels
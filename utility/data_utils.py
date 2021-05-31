import pandas as pd
import re

## Key function ##
def capitalized_words(text:str):
    '''
    str(input) -> list(output)
    Filter words(more than one letter) that are composed of uppercase letters only
    and return a list of them
    '''
    text = ' '.join(text.split()).replace('_', '')
    result = [x.strip() for x in re.findall(r'\b[A-Z][A-Z]+\b', text)]
    return result

def underbar(pairs: list):
    '''
    list(input) -> list(output)
    Add underbars to entities that lost underbars
    '''
    pairs[0] = pairs[0].replace('SIDEEFFECT',
                                'SIDE_EFFECT').replace('BIOLOGICALPROCESS',
                                                       'BIOLOGICAL_PROCESS').replace('MOLEFUNCTION',
                                                                                     'MOLE_FUNCTION')
    pairs[1] = pairs[1].replace('SIDEEFFECT',
                                'SIDE_EFFECT').replace('BIOLOGICALPROCESS',
                                                       'BIOLOGICAL_PROCESS').replace('MOLEFUNCTION',
                                                                                     'MOLE_FUNCTION')
    return pairs

def chosen_entity(pairs: list):
    '''
    list(input) -> list(output)
    Out of the uppercase composed words, leave only necessary entities
    '''
    ent = ['CELL', 'GENE', 'PHENOTYPE', 'MOLEFUNCTION', 
    'METABOLITE', 'SIDEEFFECT', 'COMPOUND', 'FINDINGS', 'BIOLOGICALPROCESS', 
    'SYMPTOMS']
    result = [p for p in pairs if p in ent]
    return result

def html_decode(text: str):
    '''
    str(input) -> str(output)
    change character entity reference to symbols
    '''
    text = text.replace('&amp;', '&').replace('&gt;', '>').replace('&lt;', '<').replace('&quot;', '"').replace('&apos;', "'")
    return text

def drop_sents(df):
    # These indexes are based on our data
    drop_idx = [71362, 56196, 74505, 74346, 62985, 59216, 66513, 48753, 69394, 51668, 67764, 62586, 
                72061, 1180, 5472, 8774, 13793, 16333, 17847, 18890, 20972, 22197, 25941, 
                33466, 34603, 36664, 37861, 38306, 41932, 41982, 86225, 88666, 88708, 
                89224, 92579, 93278, 98144, 99403, 101182, 105066, 107655, 110295, 115555, 
                119113, 119701, 122337, 122400, 122695, 124495]
    df = df.drop(drop_idx, axis=0)
    return df

## Data function ##
def extract_entity(df, sent_col):
    '''
    df, col_name(input) -> df(output)
    input the df and sentence column name to extract sentences with only 2 entities.
    Extract each entity into columns and return the resulting dataframe
    '''
    df['entity_pairs'] = df[sent_col].apply(lambda x: capitalized_words(x))  # Find capitalized words only
    df['entity_pairs'] = df.entity_pairs.apply(lambda x: chosen_entity(x))  # Choose only necessary entities
    df['count'] = df.entity_pairs.apply(lambda x: len(x))  # Create column for number of entities
    df2 = df[df['count']==2]  # Pick rows with only 2 entities

    df2['entity_pairs'] = df2.entity_pairs.apply(lambda x: underbar(x))  # Add underbar
    df2['entity1'] = df2.entity_pairs.apply(lambda x: x[0])  # 1st entity
    df2['entity2'] = df2.entity_pairs.apply(lambda x: x[1])  # 2nd entity

    df2 = df2.reset_index().drop('level_0', axis=1)  # Reset index
    df_final = df2[['text', 'answer', 'entity_pairs', 'entity1', 'entity2']]  # Choose only columns that are needed
    
    return df_final

def text_refine(df):
    '''
    df(input) -> df(output)
    Remove redundant symbols in the front and end of sentence
    Change html symbols into symbols
    '''
    df['text'] = df.text.apply(lambda x: x.strip('[').strip('.').strip(']'))  # strip [].
    df['text'] = df.text.apply(lambda x: ' '.join(x.split())+'.')  # remove redundant space between words
    
    df['text'] = df['text'].apply(lambda x: html_decode(x))
    return df

def predicate_instance_filter(df, answer_df, num: int):
    '''
    df, df, int(input) -> df, df(output)
    Filter predicates with certain number(int) of instance cases
    Update answer data accordingly as well.
    '''
    # update training data
    answer_freq = df.answer.value_counts()
    predicate_lst = [i for i in answer_freq.index.tolist() if answer_freq[i] >= num]  # more than certain number
    
    df2 = df[df['answer'].isin(predicate_lst)]
    df2 = df2[['text', 'answer']]  # leave columns that are needed
    df2 = df2.reset_index().drop('index', axis=1)
    df3 = drop_sents(df2)  # drop more than 3 entities
    
    # update answer(predicates) accordingly
    answer_df2 = answer_df[answer_df['index'].isin(predicate_lst)]
    answer_df2 = answer_df2.reset_index().drop('level_0', axis=1)
    return df3, answer_df2
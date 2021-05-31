import pandas as pd
from itertools import product
from sklearn.preprocessing import LabelEncoder

## Key Function ##
def split_neg_preds(df):
    '''
    df (input) -> df1, df2 (output)
    Input label dataframe and split the dataframe into two.
    One has predicates that don't have the prefix 'neg_'
    Another has predicates with the prefix 'neg_'
    '''
    # negative labels
    label_negs = ['neg_' + df.loc[i, 'label'] for i in range(len(df)) if 'neg_'+df.loc[i, 'label'] in df['label'].tolist()]
    # label number of negative predicate
    label_negs_num = [df.loc[i, 'index'] for i in range(len(df)) if df.loc[i, 'label'] in label_negs]
    # find index of negative predicate
    drop_idx = df[df['index'].isin(label_negs_num)].index.tolist()
    
    # make two dataframes(one with no negative predicates / only negative predicates)
    df_pos = df.drop(drop_idx, axis=0)
    df_pos = df_pos.reset_index().drop('level_0', axis=1)  # reset index

    df_neg = df[df['index'].isin(label_negs_num)]
    df_neg = df_neg.reset_index().drop('level_0', axis=1)  # reset index
    
    return df_pos, df_neg

def assign_neg_prefix(df, groups):
    '''
    df, list (input) -> df1, df2 (output)
    Input label dataframe and a list of framenet groups(only for the df_pos)
    The output will be each dataframe(with or without prefix 'neg_')
    '''
    df_pos, df_neg = split_neg_preds(df)
    df_pos['group'] = groups

    # assign the same group to those that have prefix 'neg_'
    pair = [(i, j) for i, j in product(range(len(df_neg)), range(len(df_pos)))]
    for p in pair:
        if df_neg.loc[p[0], 'label'][4:] == df_pos.loc[p[1], 'label']:
            df_neg.loc[p[0], 'group'] = df_pos.loc[p[1], 'group']
    
    return df_pos, df_neg

def return_group():
    # assign each predicate with framenet group
    groups = ['CONNECTIONS', 'PART_WHOLE', 'PART_WHOLE', 'MEDICAL_INTERVENTIONS',
            'CONNECTIONS', 'MEDICAL_INTERVENTIONS', 'SIMULTANEITY', 'PART_WHOLE', 
            'MEDICAL_INTERVENTIONS', 'CONNECTIONS', 'CONDITION_SYMPTOM_RELATION', 
            'CONDITION_SYMPTOM_RELATION', 'CONDITION_SYMPTOM_RELATION', 
            'MEDICAL_INTERVENTIONS', 'CONDITION_SYMPTOM_RELATION', 'CONDITION_SYMPTOM_RELATION', 
            'BIOLOGICAL_MECHANISM', 'BIOLOGICAL_MECHANISM', 'BIOLOGICAL_MECHANISM', 
            'BIOLOGICAL_MECHANISM', 'COMPARISON', 'MEDICAL_INTERVENTIONS', 'CONNECTIONS', 
            'CONDITION_SYMPTOM_RELATION', 'CONDITION_SYMPTOM_RELATION', 'COMPARISON']
    return groups

def label_encode(df):
    # label encode group name
    le = LabelEncoder()
    df['group'] = le.fit_transform(df['group'])
    df = df.rename(columns={'answer': 'predicate_answer', 'group': 'framenet_answer'})  # change column name
    return df


## Data Function ##
def assign_frame(answer_df, sent_df):
    '''
    df, list (input) -> df(output)
    Input label dataframe and a list of framenet groups(only for the df_pos)
    The output will the final label dataframe with framenet group column
    '''
    groups = return_group()
    df_pos, df_neg = assign_neg_prefix(answer_df, groups)
    
    # final label dataframe
    answer_fin = pd.concat([df_pos, df_neg]).sort_values('index').reset_index().drop('level_0', axis=1)
    answer_fin = answer_fin.rename(columns={"index": "answer"})

    # merge with sentence_df
    sent_df = sent_df.reset_index()
    df_fin = pd.merge(sent_df, answer_fin[['answer', 'group']], on='answer')
    df_fin = df_fin.sort_values(by='index').reset_index().drop('level_0', axis=1)  # sort values like original

    # label encode
    df_fin2 = label_encode(df_fin)
    df_fin2 = df_fin2.drop('index', axis=1)

    return df_fin2
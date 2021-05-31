import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from utility.data_utils import extract_entity, text_refine, predicate_instance_filter
from utility.framenet import assign_frame

if __name__ == '__main__':
    # read data
    data = pd.read_csv('./BioPREP/predicate_train.csv')
    answer = pd.read_csv('./BioPREP/predicate_answers.csv')

    # preprocessing
    entity_data = extract_entity(data, 'text')
    entity_data = text_refine(entity_data)
    final_data, final_answer = predicate_instance_filter(entity_data, answer, 50)
    # save answer file
    final_answer.to_csv('./BioPREP/answers.csv', index=False)
    
    # framenet clustering
    final_answer['label'] = final_answer['label'].apply(lambda x: x.lower())
    train_data = assign_frame(final_answer, final_data)
    # save train data
    train_data.to_csv('./BioPREP/train.csv', index=False)
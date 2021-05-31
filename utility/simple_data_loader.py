import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_text_label_pairs(data_file_path):
    file = open(data_file_path, mode='rt', encoding='utf8')
    next(file)  # skip header
    result = []
    for line in file:
        lst = line.strip().split(',')
        sentence = lst[0]
        label = lst[1]
        result.append((sentence, label))
    return result

def load_csv_dataset(data_file_path, label_type='predicate'):
    df = pd.read_csv(data_file_path)

    print('Brief overview on Dataset...')
    print(df.head())

    result = []
    if label_type == 'Predicate':
        print('Extracting Labels from predicate answers...')
        for idx, row in df.iterrows():
            result.append((row['text'], row['predicate_answer']))

    elif label_type == 'FrameNet':
        print('Extracting Labels from framenet answers...')
        for idx, row in df.iterrows():
            result.append((row['text'], row['framenet_answer']))

    return result

def load_bert_data(data_file_path, label_type): # label_type = "predicate" or "framenet"
    # Load data for bert model
    df = pd.read_csv(data_file_path)

    print('=====An Overview of Dataset=====')
    print(df.head())

    X = df.text.values

    label_encoder = LabelEncoder()

    if label_type == 'predicate':
        y = label_encoder.fit_transform(df.predicate_answer)
        num_classes = df.predicate_answer.nunique()
    
    elif label_type == 'framenet':
        y = label_encoder.fit_transform(df.framenet_answer)
        num_classes = df.framenet_answer.nunique()

    return X, y, num_classes


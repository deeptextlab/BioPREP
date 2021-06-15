import pandas as pd
import requests
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

    print('===== Brief Overview of Dataset =====')
    print(df.head())

    result = []
    if label_type.lower() == 'predicate':
        print('Extracting Labels from predicate answers...')
        for idx, row in df.iterrows():
            result.append((row['text'], row['predicate_answer']))

    elif label_type.lower() == 'framenet':
        print('Extracting Labels from framenet answers...')
        for idx, row in df.iterrows():
            result.append((row['text'], row['framenet_answer']))

    else:
        raise Exception('Argument LABEL_TYPE should be selected between predicate and framenet.')

    return result

def load_bert_data(data_file_path, label_type='predicate'): # label_type = "predicate" or "framenet"
    # Load data for bert model
    df = pd.read_csv(data_file_path)

    print('===== Brief Overview of Dataset =====')
    print(df.head())

    X = df.text.values

    label_encoder = LabelEncoder()
    if label_type.lower() == 'predicate':
        y = label_encoder.fit_transform(df.predicate_answer)
        num_classes = df.predicate_answer.nunique()

    elif label_type.lower() == 'framenet':
        y = label_encoder.fit_transform(df.framenet_answer)
        num_classes = df.framenet_answer.nunique()

    else:
        raise Exception('Argument LABEL_TYPE should be selected between predicate and framenet.')

    return X, y, num_classes


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

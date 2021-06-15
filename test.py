## Library
# base
import argparse
import os
import sys
import random
import warnings
warnings.filterwarnings('ignore')

# data manipulation
import numpy as np
from sklearn.model_selection import train_test_split

# tools
import torch

# user-made
from berts import BERT_for_classification
from utility.simple_data_loader import load_bert_data, download_file_from_google_drive

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='BioBERT',
                        choices=['BERT_base', 'SciBERT', 'BioBERT', 'CNN', 'MC_CNN', 'LSTM', 'BiLSTM', 'CNN_LSTM'],
                        help='default BioBERT. Arg:model_type should be selected among certain options; \
                         BERT_base, SciBert, BioBERT, CNN, MC_CNN, LSTM, BiLSTM and CNN_LSTM')

    parser.add_argument('--label-type', type=str, default='Predicate',
                        choices=['Predicate', 'FrameNet'],
                        help='default Predicate. Arg:label_type can be selected between Predicate and FrameNet')

    parser.add_argument('--seed', type=int, default=42,
                        help='fixed random seed for reproducibility. default 42')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='default 16 for BERT-based models. better to use more than 64 for CNN and BiLSTM.')

    parser.add_argument('--max-len', type=float, default=512,
                        help='default 512. Set maximum token length for input')

    parser.add_argument('--model-dir-path', type=str, default='./models',
                        help='default directory was set to models folder. \
                        Write down another path if you want to load model in different directory')

    parser.add_argument('--data_file_path', type=str, default='./BioPREP/train.csv',
                        help='default data directory was set to train file in BioPREP folder. \
                        Write down another path if you want to load your own data')

    config = parser.parse_args()
    return config

## Main Function
def main(config):

    model_type = config.model_type.lower()

    # Set directory
    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir != '' else '.'

    # Enter unlaballed dataset you want to infer.
    if config.data_file_path is None:
        data_file_path = current_dir + '/BioPREP/train.csv'
    else:
        data_file_path = config.data_file_path

    # Load data and Split into Train/dev set with given test_size
    X, y, num_labels = load_bert_data(data_file_path, label_type=config.label_type)

    X_train, X_test = train_test_split(X, random_state=config.seed, stratify=y)

    print('===========================================')
    print('Below is the shape of test dataset.')
    print('===========================================')
    print(X_test.shape)
    print('===========================================')

    if config.label_type == 'Predicate':
        if config.model_dir_path == './models':
            # Sample fine-tuned model based on BioBERT. Download .pth file using Google Drive
            file_id = '1MVD1irqedK0K7DERBt3gtQqT3snUWWFf'
            model_file_name = config.model_dir_path + '/BioBERT_0107_09_20_20_0.8462.pth'
            
            if not os.path.exists('./models'):
                os.mkdir('./models')

            print('Downloading model..')
            print('===========================================')
            download_file_from_google_drive(file_id, model_file_name)
        else:
            model_file_name = config.model_dir_path + 'yourmodelname.pth'
    else:
        if config.model_dir_path == './models':
            file_id = '1xOzYCWDsFr789EaLIWiOfD5S_K2YTw7q'
            model_file_name = config.model_dir_path + '/BioBERT_FNet_0106_09_03_20_0.8783.pth'
            if not os.path.exists('./models'):
                os.mkdir('./models')
            print('Downloading model..')
            print('===========================================')
            download_file_from_google_drive(file_id, model_file_name)
        else:
            model_file_name = config.model_dir_path + 'yourmodelname.pth'
    print('===== Download finished')

    # Infer unlabelled dataset using loaded fine-tuned model.
    # Results would be saved in prediction folder.
    model = BERT_for_classification(model_type, num_labels)
    model.pred(X_test, model_file_name, model_type)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
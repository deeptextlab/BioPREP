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
from utility.simple_data_loader import load_bert_data, load_csv_dataset
from utility.text_fit import fit_text

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

    parser.add_argument('--epochs', type=int, default=20,
                        help='default 20')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='default 16 for BERT-based models. better to use more than 64 for CNN and BiLSTM.')

    parser.add_argument('--test_size', type=float, default=0.2,
                        help='default 0.2. Set ratio for train/dev data split.')

    parser.add_argument('--max-len', type=float, default=512,
                        help='default 512. Set maximum token length for input')

    parser.add_argument('--lr', type=float, default=5e-5,
                        help='default 5e-5. Set learning rate for AdamW optimizer.')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='default 1e-8. Set epsilon value for AdamW optimizer.')

    parser.add_argument('--eval_interval', type=int, default=5,
                        help='default 5, with default epoch 20. You can determine how many times \
                        the evaluation on the training process will happen.')

    parser.add_argument('--output-dir-path', type=str, default='/models',
                        help='default directory was set to models folder. \
                        Write down another path if you want to save model in different directory')

    parser.add_argument('--data_file_path', type=str, default='/BioPREP/train.csv',
                        help='default data directory was set to train file in BioPREP folder. \
                        Write down another path if you want to load your own data')

    config = parser.parse_args()
    return config


## Main Function
def main(config):

    '''
    Trains model using given model type and saves it.
    '''

    # convert all arguments to lowercase
    model_type, label_type = config.model_type.lower(), config.label_type.lower()

    # for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)  # in case of using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set directory
    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir != '' else '.'

    # Output directory
    output_dir_path = current_dir + config.output_dir_path
    data_file_path = current_dir + config.data_file_path

    print(f'\n\nOutput directory path for saving model <--- {output_dir_path}')
    print(f'Data file path for loading training data <--- {data_file_path}')

    # Load data and Generate model following given model_type
    if model_type in ['bert_base', 'scibert', 'biobert']:

        # Load data and Split into Train/dev set with given test_size
        X, y, num_labels = load_bert_data(data_file_path, label_type=label_type)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size, random_state=config.seed, stratify=y)

        print('\n\n===========================================')
        print('Below is the shape of train/test dataset.')
        print('===========================================')
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        print('===========================================\n\n')

        # Generate model
        model = BERT_for_classification(model_type, num_labels)
        model.fit(train_data = (X_train, y_train),
                  test_data = (X_test, y_test),
                  batch_size=config.batch_size,
                  epochs=config.epochs,
                  max_len=config.max_len,
                  test_size=config.test_size,
                  seed=config.seed,
                  lr=config.lr,
                  eps=config.eps,
                  eval_interval=config.eval_interval)

        # Draw and save a precision-recall curve only for FrameNet labels
        if label_type == 'framenet':
            model.plot()

    elif model_type in ['cnn', 'mc_cnn', 'lstm', 'bilstm', 'cnn_lstm']:
        text_data_model = fit_text(data_file_path, label_type=label_type)
        text_label_pairs = load_csv_dataset(data_file_path, label_type=label_type)

        if model_type == 'cnn':
            from cnn import WordVecCnn
            classifier = WordVecCnn()
        elif model_type == 'mc_cnn':
            from cnn import WordVecMultiChannelCnn
            classifier = WordVecMultiChannelCnn()
        elif model_type == 'lstm':
            from lstm import WordVecLstmSoftmax
            classifier = WordVecLstmSoftmax()
        elif model_type == 'bilstm':
            from lstm import WordVecBidirectionalLstmSoftmax
            classifier = WordVecBidirectionalLstmSoftmax()
        elif model_type == 'cnn_lstm':
            from cnn_lstm import WordVecCnnLstm
            classifier = WordVecCnnLstm()

        history = classifier.fit(text_data_model=text_data_model,
                                 model_dir_path=output_dir_path,
                                 text_label_pairs=text_label_pairs,
                                 batch_size=config.batch_size,
                                 epochs=config.epochs,
                                 test_size=config.test_size,
                                 random_state=config.seed)


if __name__ == '__main__':
    config = define_argparser()
    main(config)

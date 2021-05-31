import numpy as np
import sys
import os


def main(NN_type:str, label_type:str): # label_type -> 'Predicate' or 'FrameNet'
    random_state = 42
    np.random.seed(random_state)

    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir != '' else '.'

    output_dir_path = current_dir + '/models'
    data_file_path = current_dir + '/BioPREP/train.csv'

    from utility.simple_data_loader import load_text_label_pairs, load_csv_dataset
    from utility.text_fit import fit_text

    text_data_model = fit_text(data_file_path, label_type=label_type)
    text_label_pairs = load_csv_dataset(data_file_path, label_type=label_type)

    if NN_type == 'CNN':
        from cnn import WordVecCnn
        classifier = WordVecCnn()
    elif NN_type == 'MC_CNN':
        from cnn import WordVecMultiChannelCnn
        classifier = WordVecMultiChannelCnn()
    elif NN_type == 'LSTM':
        from lstm import WordVecLstmSoftmax
        classifier = WordVecLstmSoftmax()
    elif NN_type == 'BiLSTM':
        from lstm import WordVecBidirectionalLstmSoftmax
        classifier = WordVecBidirectionalLstmSoftmax()
    elif NN_type == 'CNN_LSTM':
        from cnn_lstm import WordVecCnnLstm
        classifier = WordVecCnnLstm()
    else:
        raise Exception('Please enter correct NN types!')

    batch_size = 64
    epochs = 20

    history = classifier.fit(text_data_model=text_data_model,
                             model_dir_path=output_dir_path,
                             text_label_pairs=text_label_pairs,
                             batch_size=batch_size,
                             epochs=epochs,
                             test_size=0.2,
                             random_state=random_state)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

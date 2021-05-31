import collections

from utility.tokenizer_utils import word_tokenize


def fit_text(data_file_path, label_type, max_vocab_size=None):
    if max_vocab_size is None:
        max_vocab_size = 5000

    counter = collections.Counter()
    file = open(data_file_path, mode='rt', encoding='utf8')
    next(file) # skip header
    max_len = 0
    labels = dict()

    for line in file:
        lst = line.strip().split(',')
        sentence = lst[0]

        if label_type == 'Predicate':
            label = lst[1]
        elif label_type == 'FrameNet':
            label = lst[2]

        # tokens = [x.lower() for x in word_tokenize(sentence)]
        tokens = [x for x in word_tokenize(sentence)] # Cased word 유지
        for token in tokens:
            counter[token] += 1
        max_len = max(max_len, len(tokens))

        if label not in labels: # 라벨에 0부터 번호 부여(라벨당 개수 세는 것이 아님)
            labels[label] = len(labels)
    file.close()

    word2idx = collections.defaultdict(int)
    for idx, word in enumerate(counter.most_common(max_vocab_size)):
        word2idx[word[0]] = idx
    idx2word = {v: k for k, v in word2idx.items()}
    vocab_size = len(word2idx) + 1

    model = dict()

    model['word2idx'] = word2idx
    model['idx2word'] = idx2word
    model['vocab_size'] = vocab_size
    model['max_len'] = max_len
    model['labels'] = labels

    return model

import re
import gensim
import numpy as np
import torch


class Argurment():
    def __init__(self):
        self.n_layer = 2
        self.num_classes = 200
        self.embedding_dim = 300
        self.hidden_size = 256
        self.filter_size = 256
        self.STR_UNKNOWN = '<unknown>'
        path = '/home/phivantuan/Documents/tiki/'
        self.model_word2vec = gensim.models.KeyedVectors.load_word2vec_format(path + 'vector_merge_300.bin',
                                                                              binary=True)
        array_unkonwn = np.zeros(self.embedding_dim)
        self.model_word2vec.add(self.STR_UNKNOWN, array_unkonwn)
        self.dropout = 0.5
        self.weights = torch.FloatTensor(self.model_word2vec.wv.vectors)
        self.batch_size = 60

    def pre_process(self, text):
        text = text.lower()
        # text=text.translate(str.maketrans(' ', ' ', string.punctuation))
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' <number>', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub('\s+', ' ', text)

        return text.split()

    def word2idx(self, word):
        if word in self.model_word2vec.wv.vocab:return self.model_word2vec.vocab[word].index
        else:return self.model_word2vec.vocab[self.STR_UNKNOWN].index

    def pad_features(self, input, seq_length=50):

        review = self.pre_process(input)
        lenth = len(review)
        array = np.zeros(seq_length)
        if lenth > seq_length: review = review[-seq_length:]
        for index, word in enumerate(review):
            array[index] = self.word2idx(word)
        return array


    def idx2word(self, idx):
        return self.model_word2vec.index2word[idx]

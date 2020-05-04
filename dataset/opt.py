import re
import gensim
import numpy as np
import torch
from numpy.linalg import norm


class Argurment():
    def __init__(self):
        self.n_layer = 2
        self.num_classes = 512
        self.embedding_dim = 300
        self.hidden_size = 256
        self.filter_size = 256
        self.STR_UNKNOWN = '<unknown>'
        self.STR_PADDING = '<padding>'
        self.path = '/home/phivantuan/Documents/tiki/a/'
        self.path2 = '/home/phivantuan/Documents/tiki/'
        self.model_word2vec = gensim.models.KeyedVectors.load_word2vec_format(self.path2 + 'vector_merge_300.bin',
                                                                              binary=True)
        array_unkonwn = np.zeros(self.embedding_dim)

        self.model_word2vec.add(self.STR_UNKNOWN, array_unkonwn)
        # self.model_word2vec.add(self.STR_PADDING, array_unkonwn)
        self.length_vocab = len(self.model_word2vec.vocab)
        self.dropout = 0.5
        self.weights = torch.FloatTensor(self.model_word2vec.wv.vectors)
        self.batch_size = 50

    def pre_process(self, text):
        text = text.lower()
        # text=text.translate(str.maketrans(' ', ' ', string.punctuation))
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' <number> ', text)
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

    def vector_word(self,word):
        if word in self.model_word2vec.wv.vocab:return self.model_word2vec[word]
        else:return self.model_word2vec[self.STR_UNKNOWN]
    def matrix_(self,file):
        result = []
        array = []

        for line in file:
            result.append([])
            x = np.array(line).astype(np.float)
            array.append(x)
            for i in range(len(array)):
                value = array[i]
                if (i + 1 == len(array)):
                    data = 1.0
                else:
                    data = np.dot(x, value) / (norm(value) * norm(x))
                    result[len(result) - 1].append(data)
                    result[i].append(data)
        return np.array(result)

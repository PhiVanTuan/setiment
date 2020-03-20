import gensim
import torch.nn as nn
import torch
import numpy as np

from torchtext.vocab import Vectors

model = gensim.models.KeyedVectors.load_word2vec_format(
    '/home/phivantuan/PycharmProjects/source-archive/word2vec/trunk/vector_merge_300.bin', binary=True)

weights = torch.FloatTensor(model.wv.vectors)


def average(arr, batch_size, hidden_dim):
    result = np.zeros((batch_size, hidden_dim))
    for index, x in enumerate(arr):
        result[index] = np.average(x)
    return result


def min_pool(arr, batch_size, hidden_dim):
    result = np.zeros((batch_size, hidden_dim))
    for index, x in enumerate(arr):
        result[index] = numpy_min(x)
    return result


def maxpool(arr, batch_size, hidden_dim):
    result = np.zeros((batch_size, hidden_dim))
    for index, x in enumerate(arr):
        result[index] = numpy_max(x)
    return result


def numpy_min(arr):
    xresult = arr[0]
    for x in arr:
        xresult = np.minimum(x, xresult)
    return xresult


def numpy_max(arr):
    xresult = arr[0]
    for x in arr:
        xresult = np.maximum(x, xresult)
    return xresult


class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding.from_pretrained(weights)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)
        self.FC = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)
        # input=x.view(batch_size,100,300)
        # print(hidden)
        # embeds = self.embedding(x)
        embeds = self.embedding(x)

        # print(embeds.shape)
        # embeds = embeds.view(batch_size,100,300)
        lstm_out, hidden = self.lstm(embeds, hidden)
        # result=lstm_out.detach().numpy()
        # result=maxpool(result,batch_size,self.hidden_dim)
        # result=torch.Tensor(result)
        # stack_up lstm outputs

        # lstm_out = result.contiguous().view(-1, self.hidden_dim)
        # print(lstm_out.shape)
        # out = self.FC(lstm_out)
        out = self.dropout(lstm_out)
        out = self.FC(out[:, -1, :])

        # return last sigmoid output and hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        train_on_gpu = torch.cuda.is_available()
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden
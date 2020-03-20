import re
import string
from collections import Counter
from numpy.linalg import norm
import gensim
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchtext import data

from muse.lstm import SentimentRNN

batch_size = 40
model = gensim.models.KeyedVectors.load_word2vec_format(
    '/home/phivantuan/PycharmProjects/source-archive/word2vec/trunk/vector_merge_300.bin', binary=True)


def pre_process(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' <number>', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text.split()


def word2idx(word):
    return model.vocab[word].index


def idx2word(idx):
    return model.index2word[idx]


def pad_features(input, seq_length=50):
    array = np.zeros(seq_length)
    review = pre_process(input)
    lenth = len(review)
    if lenth < seq_length:
        for index, word in enumerate(review):
            if word in model.wv.vocab:
                array[seq_length - lenth + index] = word2idx(word)
    elif lenth > seq_length:
        review = review[-seq_length:]
        for index, word in enumerate(review):
            if word in model.wv.vocab:
                array[index] = word2idx(word)
    else:
        for index, word in enumerate(review):
            if word in model.wv.vocab:
                array[index] = word2idx(word)
    return array


def matrix_(file):
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


def pre_data():
    with open('/home/phivantuan/Documents/vn_use/train_sens.txt') as in_file:
        features = [pad_features(x) for x in in_file]
    encoded_labels = np.arange(start=0, stop=len(features), step=1)

    # print(features[0])
    ## split data into training, validation, and test data (features and labels, x and y)
    train_x, remaining_x, train_y, remaining_y = train_test_split(features, encoded_labels, test_size=0.1)
    test_x, valid_x, test_y, valid_y = train_test_split(remaining_x, remaining_y, test_size=0.5)
    print("features " + str(len(train_x)) + "  " + str(len(valid_x)))
    train_data = TensorDataset(torch.LongTensor(train_x), torch.as_tensor(np.array(train_y).astype('long')))
    valid_data = TensorDataset(torch.LongTensor(valid_x), torch.as_tensor(np.array(valid_y).astype('long')))
    # # dataloaders
    # # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    # print(train_loader)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader, valid_loader


output_size = 512
embedding_dim = 300
hidden_dim = 256
n_layers = 2

net = SentimentRNN(output_size, embedding_dim, hidden_dim, n_layers)
train_on_gpu = torch.cuda.is_available()


def train():
    lr = 0.0001

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    epochs = 5  # 3-4 is approx where I noticed the validation loss stop decreasing

    counter = 0
    print_every = 100
    clip = 5  # gradient clipping
    train_loader, valid_loader = pre_data()
    print(str(len(train_loader)) + "   :   " + str(len(valid_loader)))
    # move model to GPU, if available
    if (train_on_gpu):
        net.cuda()

    net.train()
    # train for some number of epochs
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)

        # batch loop
        for inputs, labels in train_loader:
            counter += 1
            inputs = inputs.type(torch.LongTensor)
            if (train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])
            # zero accumulated gradients
            net.zero_grad()
            # print(inputs)
            # get the output from the model
            output, h = net(inputs, h)
            # calculate the loss and perform backprop
            result = matrix_(output.detach().numpy())
            label = labels.detach().numpy()
            array = []
            for index in label:
                path = '/home/phivantuan/Documents/translate/' + str(index) + ".txt"
                with open(path) as label_file:
                    array.append(np.array(label_file.read().split()).astype(float))
            label = matrix_(array)

            loss = criterion(torch.tensor(result, requires_grad=True), torch.tensor(label, requires_grad=True))
            print(str(counter) + "   :    " + str(loss))
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()
            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                sums = []
                sizes = []
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for inputs, labels in valid_loader:
                    inputs = inputs.type(torch.LongTensor)
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    if (train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output, val_h = net(inputs, val_h)
                    result = matrix_(output.detach().numpy())
                    label = labels.detach().numpy()
                    array = []
                    for index in label:
                        path = '/home/phivantuan/Documents/translate/' + str(index) + ".txt"
                        with open(path) as label_file:
                            array.append(np.array(label_file.read().split()).astype(float))
                    label = matrix_(array)

                    val_loss = criterion(torch.tensor(result, requires_grad=True),
                                         torch.tensor(label, requires_grad=True))

                    val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))


train()

torch.save(net.state_dict(), "lstmmodel")

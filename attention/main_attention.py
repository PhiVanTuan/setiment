import re
from _ctypes import ArgumentError

import gensim
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from review.dataset import Dataset
from attention.model import AttentionModel
from dataset.opt import Argurment
from attention.model_bilstm_attention import LSTMAttention
# STR_UNKNOWN='<unknown>'
path='/home/phivantuan/Documents/tiki/'
# model_word2vec = gensim.models.KeyedVectors.load_word2vec_format(path+'vector_merge_300.bin', binary=True)
# array_unkonwn=np.zeros(300)
# model_word2vec.add(STR_UNKNOWN,array_unkonwn)

# weights = torch.FloatTensor(model_word2vec.wv.vectors)

num_epoch = 10
learning_rate = 0.0001
loss_function = nn.CrossEntropyLoss()
train_on_gpu = torch.cuda.is_available()
argurment=Argurment()
model = LSTMAttention(argurment)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# def pre_process(text):
#     text = text.lower()
#     # text=text.translate(str.maketrans(' ', ' ', string.punctuation))
#     text = re.sub(r'[^\w\s]', ' ', text)
#     text = re.sub(r'\d+', ' <number>', text)
#     text = re.sub(r'\n', ' ', text)
#     text = re.sub('\s+', ' ', text)
#
#     return text.split()
#
#
# def word2idx(word):
#     if word in model_word2vec.wv.vocab: return model_word2vec.vocab[word].index
#     else : return model_word2vec.vocab[STR_UNKNOWN].index
#
#
# def idx2word(idx):
#     return model_word2vec.index2word[idx]
#
#
# def pad_features(input, seq_length=50):
#     array = np.zeros(seq_length)
#     review = pre_process(input)
#     lenth = len(review)
#     if lenth > seq_length:review = review[-seq_length:]
#     for index, word in enumerate(review):
#         array[index] = word2idx(word)
#
#     return array


def pre_data():
    features = open(path + 'review_train.txt').read().splitlines()
    encoded_labels = [int(i) for i in open(path + 'label_train.txt').read().splitlines()]
    valid_x = open(path + 'review_valid.txt').read().splitlines()
    valid_y = [int(i) for i in open(path + 'label_valid.txt').read().splitlines()]
    train_data = Dataset(features, torch.LongTensor(encoded_labels))
    valid_data = Dataset(valid_x, torch.LongTensor(valid_y))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=argurment.batch_size, drop_last=True)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=argurment.batch_size, drop_last=True)
    return train_loader, valid_loader


def train():
    counter = 0
    print_every = 100
    clip = 5  # gradient clipping
    train_loader, valid_loader = pre_data()
    num_batch=len(train_loader)
    if (train_on_gpu): model.cuda()
    model.train()
    for i in range(num_epoch):
        sum_loss=0
        for inputs, labels in train_loader:
            counter += 1
            inputs = list(inputs)
            inputs = [argurment.pad_features(x) for x in inputs]
            inputs = torch.LongTensor(inputs)
            inputs = inputs.type(torch.LongTensor)
            if (train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()
            # h = tuple([each.data for each in h])
            model.zero_grad()
            # print(inputs)
            # get the output from the model
            output = model(inputs)

            # calculate the loss and perform backprop

            loss = loss_function(output.squeeze(), labels)
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            sum_loss+=loss.item()
            print(str(counter) + " : " + str(loss.item()))
            if counter % print_every == 0:
                # val_h = model.init_hidden(batch_size)
                val_losses = []
                sums = 0
                correct = 0
                model.eval()
                for inputs, labels in valid_loader:
                    inputs = list(inputs)
                    inputs = [argurment.pad_features(x) for x in inputs]
                    inputs = torch.LongTensor(inputs)
                    inputs = inputs.type(torch.LongTensor)
                    if (train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output = model(inputs)
                    val_loss = loss_function(output.squeeze(), labels)
                    tensor_max_value, index = torch.max(output.data, 1)
                    sums += labels.size(0)
                    correct += index.eq(labels.data).sum().item()
                    predict = correct / sums * 100
                    val_losses.append(val_loss.item())

                model.train()
                print("Epoch: {}/{}...".format(i + 1, num_epoch),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(sum_loss/(counter-num_batch*i)),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)),
                      "Accuracy:{:.2f}".format(predict))

def test():
    features = [argurment.pad_features(x) for x in open(path+'review_test.txt').read().splitlines()]
    encoded_labels = [int(i) for i in open(path+'label_test.txt').read().splitlines()]
    train_data = TensorDataset(torch.as_tensor(np.array(features).astype('long')), torch.LongTensor(encoded_labels))
    train_loader = DataLoader(train_data, shuffle=False, batch_size=50, drop_last=True)

    sums = 0
    correct = 0
    model.eval()
    for inputs, labels in train_loader:


        if (train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        output = model(inputs)
        tensor_max_value, index = torch.max(output.data, 1)
        sums += labels.size(0)
        correct += index.eq(labels.data).sum().item()
        predict = correct / sums * 100
        print("Correct: {}".format(correct),
              "Sums: {}...".format(sums),
              "Accuracy:{:.2f}".format(predict))


train()

torch.save(model.state_dict(), "attention_model")

model.load_state_dict(torch.load("attention_model"))

model.eval()
#
test()
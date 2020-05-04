import re
import gensim
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from review.model_cnn import CnnTextClassifier
from review.model_cnnlstm import CNN_LSTM
from review.model_lstmcnn import LSTM_CNNTextClassifier
from review.model_lstm import SentimentRNN


def get_model(option):
    if option == LSTM:
        model = SentimentRNN(weights, num_classes, embedding_dim, hidden_dim, n_layers)
    elif option == CNN:
        model = CnnTextClassifier(weights, num_classes, num_filters)
    elif option == LSTM_CNN:
        model = LSTM_CNNTextClassifier(weights, num_classes, num_filters, embedding_dim, hidden_dim, n_layers)
    elif option == CNN_LSTM:
        model = CNN_LSTM(weights, num_classes, num_filters, embedding_dim, hidden_dim, n_layers)
    return model


LSTM = 0
CNN = 1
LSTM_CNN = 2
CNN_LSTM = 3
STR_UNKNOWN='<unknown>'
path='/home/phivantuan/Documents/tiki/'
model_word2vec = gensim.models.KeyedVectors.load_word2vec_format(path+'vector_300.bin', binary=True)
array_unkonwn=np.zeros(300)
model_word2vec.add(STR_UNKNOWN,array_unkonwn)

weights = torch.FloatTensor(model_word2vec.wv.vectors)
batch_size = 50
num_classes = 2
num_filters = 256
embedding_dim = 300
hidden_dim = 256
n_layers = 2
num_epoch = 10
learning_rate = 0.0001
loss_function = nn.CrossEntropyLoss()
train_on_gpu = torch.cuda.is_available()
model = get_model(LSTM_CNN)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def pre_process(text):
    text = text.lower()
    # text=text.translate(str.maketrans(' ', ' ', string.punctuation))
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' <number> ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub('\s+', ' ', text)

    return text.split()


def word2idx(word):
    if word in model_word2vec.wv.vocab: return model_word2vec.vocab[word].index
    else : return model_word2vec.vocab[STR_UNKNOWN].index


def idx2word(idx):
    return model_word2vec.index2word[idx]


def pad_features(input, seq_length=50):
    array = np.zeros(seq_length)
    review = pre_process(input)
    lenth = len(review)
    if lenth > seq_length:review = review[-seq_length:]
    for index, word in enumerate(review):
        array[index] = word2idx(word)

    return array


def pre_data():
    features = [pad_features(x) for x in open(path+'review_train.txt').read().splitlines()]
    encoded_labels = [int(i) for i in open(path+'label_train.txt').read().splitlines()]
    valid_x = [pad_features(x) for x in open(path+'review_valid.txt').read().splitlines()]
    valid_y = [int(i) for i in open(path+'label_valid.txt').read().splitlines()]
    train_data = TensorDataset(torch.as_tensor(np.array(features).astype('long')), torch.LongTensor(encoded_labels))
    valid_data = TensorDataset(torch.as_tensor(np.array(valid_x).astype('long')), torch.LongTensor(valid_y))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loader, valid_loader


def train():
    counter = 0
    print_every = 100
    clip = 5  # gradient clipping
    train_loader, valid_loader = pre_data()
    num_batch=len(train_loader)
    print(num_batch)
    if (train_on_gpu): model.cuda()
    model.train()
    for i in range(num_epoch):
        h = model.init_hidden(batch_size)
        sum_loss = 0
        for inputs, labels in train_loader:
            counter += 1

            if (train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()
            h = tuple([each.data for each in h])
            model.zero_grad()
            # print(inputs)
            # get the output from the model
            output, h = model(inputs, h)

            # calculate the loss and perform backprop

            loss = loss_function(output.squeeze(), labels)
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            sum_loss+=loss.item()
            print(str(counter) + " : " + str(loss.item()))
            if counter % print_every == 0:
                val_h = model.init_hidden(batch_size)
                val_losses = []
                sums=0
                correct=0
                model.eval()
                for inputs, labels in valid_loader:
                    val_h = tuple([each.data for each in val_h])
                    if (train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output, val_h = model(inputs, val_h)
                    val_loss = loss_function(output.squeeze(), labels)
                    tensor_max_value, index = torch.max(output.data, 1)
                    sums+= labels.size(0)
                    correct+= index.eq(labels.data).sum().item()
                    predict = correct / sums*100
                    val_losses.append(val_loss.item())

                model.train()
                print("Epoch: {}/{}...".format(i + 1, num_epoch),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(sum_loss/(counter-i*num_batch)),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)),
                      "Accuracy:{:.2f}".format(predict))


def test():
    features = [pad_features(x) for x in open(path+'review_test.txt').read().splitlines()]
    encoded_labels = [int(i) for i in open(path+'label_test.txt').read().splitlines()]
    train_data = TensorDataset(torch.as_tensor(np.array(features).astype('long')), torch.LongTensor(encoded_labels))
    train_loader = DataLoader(train_data, shuffle=False, batch_size=50, drop_last=True)
    val_h = model.init_hidden(50)
    sums = 0
    correct = 0
    model.eval()
    for inputs, labels in train_loader:

        val_h = tuple([each.data for each in val_h])

        if (train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        output, val_h = model(inputs, val_h)
        tensor_max_value, index = torch.max(output.data, 1)
        sums += labels.size(0)
        correct += index.eq(labels.data).sum().item()
        predict = correct / sums * 100
        print("Correct: {}".format(correct),
              "Sums: {}...".format(sums),
              "Accuracy:{:.2f}".format(predict))


def predict(model, test_review):
    features = []
    features.append(pad_features(test_review))
    input = torch.LongTensor(features)
    return model(input)


train()

torch.save(model.state_dict(), "lstm_cnn_model")

model.load_state_dict(torch.load("lstm_cnn_model"))

model.eval()
#
test()
#
# test_review='Hàng tốt .........................................'
# print(predict(model,test_review))

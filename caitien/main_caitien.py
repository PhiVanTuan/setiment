import re
import gensim
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from attention.model_bilstm_attention import LSTMAttention
from caitien.model import Model
from caitien.model_bilstm_attention_2 import LSTMAttention2
from caitien.model_cnn_2 import CnnTextClassifier2
from dataset.opt import Argurment
path_pretrain='/home/phivantuan/PycharmProjects/translation/translation/cnn_model'
argurment=Argurment()
# model2 = LSTMAttention(argurment)
model1_dict = torch.load('lstm_attention_model')
# model2.load_state_dict(torch.load('lstm_attention_model'))
model=LSTMAttention2(argurment)
model2_dict = model.state_dict()
# 1. filter out unnecessary keys
filtered_dict = {k: v for k, v in model1_dict.items() if k in model2_dict}
# 2. overwrite entries in the existing state dict
model2_dict.update(filtered_dict)
# 3. load the new state dict
model.load_state_dict(model2_dict)
num_epoch = 5
learning_rate = 0.0001

loss_function = nn.CrossEntropyLoss()
train_on_gpu = torch.cuda.is_available()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
def pre_data():
    batch_size=50
    features = [argurment.pad_features(x) for x in open(argurment.path+'review_train.txt').read().splitlines()]
    encoded_labels = [int(i) for i in open(argurment.path+'label_train.txt').read().splitlines()]
    valid_x = [argurment.pad_features(x) for x in open(argurment.path+'review_valid.txt').read().splitlines()]
    valid_y = [int(i) for i in open(argurment.path+'label_valid.txt').read().splitlines()]
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
    if (train_on_gpu): model.cuda()
    model.train()
    for i in range(num_epoch):
        sum_loss=0
        for inputs, labels in train_loader:
            counter += 1
            # inputs=model_pre_train(inputs)
            if (train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

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
            print(str(counter) + " : " + str(loss.item()))
            sum_loss+=loss.item()
            if counter % print_every == 0:

                val_losses = []
                sums = 0
                sum_neg=0
                sum_neu=0
                sum_pos=0
                correct = 0
                model.eval()
                for inputs, labels in valid_loader:
                    # inputs = model_pre_train(inputs)
                    if (train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output = model(inputs)
                    val_loss = loss_function(output.squeeze(), labels)
                    tensor_max_value, index = torch.max(output.data, 1)
                    out=labels.data
                    sums += labels.size(0)
                    correct += index.eq(labels.data).sum().item()
                    predict = correct / sums * 100
                    val_losses.append(val_loss.item())

                model.train()
                print("Epoch: {}/{}...".format(i + 1, num_epoch),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(sum_loss/(counter-i*num_batch)),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)),
                      "Accuracy:{:.2f}".format(predict))

def test():
    features = [argurment.pad_features(x) for x in open(argurment.path + 'review_test.txt').read().splitlines()]
    encoded_labels = [int(i) for i in open(argurment.path + 'label_test.txt').read().splitlines()]
    train_data = TensorDataset(torch.as_tensor(np.array(features).astype('long')), torch.LongTensor(encoded_labels))
    train_loader = DataLoader(train_data, shuffle=False, batch_size=50, drop_last=True)

    sums = 0
    correct = 0
    model.eval()
    for inputs, labels in train_loader:
        # inputs=model_pre_train(inputs)
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

torch.save(model.state_dict(), "caitien_model")

model.load_state_dict(torch.load("caitien_model"))

model.eval()

test()

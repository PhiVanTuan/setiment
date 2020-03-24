# import re
# import gensim
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
#
# from review.dataset import Dataset
# from review.model_lstm import SentimentRNN
#
# model = gensim.models.KeyedVectors.load_word2vec_format(
#     '/home/phivantuan/PycharmProjects/source-archive/word2vec/trunk/vector_merge_300.bin', binary=True)
#
#
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
#     return model.vocab[word].index
#
#
# def idx2word(idx):
#     return model.index2word[idx]
#
#
# def pad_feature(input, seq_length=100):
#     array = np.zeros((100, 300))
#     review = pre_process(input)
#     lenth = len(review)
#     if lenth < seq_length:
#         for index, word in enumerate(review):
#             if word in model.wv.vocab:
#                 vec = model[word]
#                 array[seq_length - lenth + index] = vec
#     elif lenth > seq_length:
#         review = review[-100:]
#         for index, word in enumerate(review):
#             if word in model.wv.vocab:
#                 vec = model[word]
#                 array[index] = vec
#     else:
#         for index, word in enumerate(review):
#             if word in model.wv.vocab:
#                 vec = model[word]
#                 array[index] = vec
#     return array
#
#
# def pad_features(input, seq_length=50):
#     array = np.zeros(seq_length)
#     review = pre_process(input)
#     lenth = len(review)
#     if lenth < seq_length:
#         for index, word in enumerate(review):
#             if word in model.wv.vocab:
#                 array[seq_length - lenth + index] = word2idx(word)
#     elif lenth > seq_length:
#         review = review[-seq_length:]
#         for index, word in enumerate(review):
#             if word in model.wv.vocab:
#                 array[index] = word2idx(word)
#     else:
#         for index, word in enumerate(review):
#             if word in model.wv.vocab:
#                 array[index] = word2idx(word)
#     return array
#
#
# batch_size = 60
# vocab_size = len(model.wv.vocab) + 1
# output_size = 3
# embedding_dim = 300
# hidden_dim = 256
# n_layers = 2
# net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
# train_on_gpu = torch.cuda.is_available()
#
#
# def pre_data():
#     features = open('review_train.txt').read().splitlines()
#     encoded_labels = [int(i) for i in open("label_train.txt").read().splitlines()]
#     valid_x = open('/home/phivantuan/Desktop/review_valid.txt').read().splitlines()
#     valid_y = [int(i) for i in open("/home/phivantuan/Desktop/label_valid.txt").read().splitlines()]
#     # print(features[0])
#     ## split data into training, validation, and test data (features and labels, x and y)
#     # train_x, remaining_x, train_y, remaining_y = train_test_split(features, encoded_labels, test_size=0.1)
#     # test_x, valid_x, test_y, valid_y = train_test_split(remaining_x, remaining_y, test_size=0.5)
#     # print("features " + str(len(train_x))+"  "+str(len(valid_x)))
#     train_data = Dataset(features, torch.as_tensor(np.array(encoded_labels).astype('long')))
#     valid_data = Dataset(valid_x, torch.as_tensor(np.array(valid_y).astype('long')))
#     # # dataloaders
#     # # make sure to SHUFFLE your data
#     train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
#     # print(train_loader)
#     valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last=True)
#     return train_loader, valid_loader
#
#
# def train():
#     lr = 0.0001
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(net.parameters(), lr=lr)
#     epochs = 10  # 3-4 is approx where I noticed the validation loss stop decreasing
#     sums = 0
#     correct = 0
#     counter = 0
#     print_every = 100
#     clip = 5  # gradient clipping
#     train_loader, valid_loader = pre_data()
#     print(str(len(train_loader)) + " : " + str(len(valid_loader)))
#     # move model to GPU, if available
#     if (train_on_gpu):
#         net.cuda()
#
#     net.train()
#     # train for some number of epochs
#     for e in range(epochs):
#         # initialize hidden state
#         h = net.init_hidden(batch_size)
#
#         # batch loop
#         for inputs, labels in train_loader:
#             counter += 1
#             print(counter)
#             inputs = list(inputs)
#             inputs = [pad_features(x) for x in inputs]
#             inputs = torch.as_tensor(np.array(inputs).astype('long'))
#             if (train_on_gpu):
#                 inputs, labels = inputs.cuda(), labels.cuda()
#
#             # Creating new variables for the hidden state, otherwise
#             # we'd backprop through the entire training history
#             h = tuple([each.data for each in h])
#
#             # zero accumulated gradients
#             net.zero_grad()
#             # print(inputs)
#             # get the output from the model
#             output, h = net(inputs, h)
#
#             # calculate the loss and perform backprop
#
#             loss = criterion(output.squeeze(), labels)
#             loss.backward()
#             # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
#             nn.utils.clip_grad_norm_(net.parameters(), clip)
#             optimizer.step()
#
#             # loss stats
#             if counter % print_every == 0:
#                 # Get validation loss
#
#                 val_h = net.init_hidden(batch_size)
#                 val_losses = []
#                 net.eval()
#                 for inputs, labels in valid_loader:
#                     inputs = list(inputs)
#                     inputs = [pad_features(x) for x in inputs]
#                     inputs = torch.as_tensor(np.array(inputs).astype('long'))
#                     # Creating new variables for the hidden state, otherwise
#                     # we'd backprop through the entire training history
#                     val_h = tuple([each.data for each in val_h])
#
#                     if (train_on_gpu):
#                         inputs, labels = inputs.cuda(), labels.cuda()
#
#                     output, val_h = net(inputs, val_h)
#                     val_loss = criterion(output.squeeze(), labels)
#                     tensor_max_value, index = torch.max(output.data, 1)
#                     sums += labels.size(0)
#                     correct += index.eq(labels.data).sum().item()
#                     predict = correct / sums
#                     val_losses.append(val_loss.item())
#
#                 net.train()
#                 print("Epoch: {}/{}...".format(e + 1, epochs),
#                       "Step: {}...".format(counter),
#                       "Loss: {:.6f}...".format(loss.item()),
#                       "Val Loss: {:.6f}".format(np.mean(val_losses)),
#                       "Accuracy:{:.2f}".format(predict))
#
#
# def predict(net, test_review):
#     features = []
#     features.append(pad_features(test_review))
#     # features=[pad_features(x) for x in open('review_test.txt').read().splitlines()]
#
#     # changing the features to PyTorch tensor
#     features = torch.as_tensor(np.array(features).astype('long'))
#
#     # pass the features to the model to get prediction
#     net.eval()
#     val_h = net.init_hidden(1)
#     val_h = tuple([each.data for each in val_h])
#
#     if (train_on_gpu):
#         features = features.cuda()
#
#     output, val_h = net(features, val_h)
#     tensor_max_value, index = torch.max(output.data, 1)
#     return output, index.item()
#
#
# def test():
#     features = open('review_test.txt').read().splitlines()
#     encoded_labels = [int(i) for i in open("label_test.txt").read().splitlines()]
#     train_data = Dataset(features, torch.as_tensor(np.array(encoded_labels).astype('long')))
#     train_loader = DataLoader(train_data, shuffle=False, batch_size=50, drop_last=True)
#     val_h = net.init_hidden(50)
#     sums = 0
#     correct = 0
#     net.eval()
#     for inputs, labels in train_loader:
#         inputs = list(inputs)
#         inputs = [pad_features(x) for x in inputs]
#         inputs = torch.as_tensor(np.array(inputs).astype('long'))
#         # Creating new variables for the hidden state, otherwise
#         # we'd backprop through the entire training history
#         val_h = tuple([each.data for each in val_h])
#
#         if (train_on_gpu):
#             inputs, labels = inputs.cuda(), labels.cuda()
#
#         output, val_h = net(inputs, val_h)
#         tensor_max_value, index = torch.max(output.data, 1)
#         sums += labels.size(0)
#         correct += index.eq(labels.data).sum().item()
#         predict = correct / sums
#         print("Correct: {}".format(correct),
#               "Sums: {}...".format(sums),
#               "Accuracy:{:.2f}".format(predict))
#
#
# # train()
# #
# # torch.save(net.state_dict(), "lstmmodel2")
#
# net.load_state_dict(torch.load("lstmmodel2"))
#
# net.eval()
#
# test()
# # test()
# # test()
# # file=open('review_train.txt').read().splitlines()
# # label=open('label_train.txt').read().splitlines()
# # total=len(file)
# # result=0
# # for index,line in enumerate(file):
# #     label_value=label[index]
# #     label_predict=predict(net,line)
# #     if str(label_value)==str(label_predict):result+=1
# #     else:print(str(label_predict)+" : "+str(label_value)+" : "+str(index+1))
# #
# # print(result/total)
#
# # test_review = "rất hay và thú vị"
# # print(predict(net, test_review))
# # print(test_review)
#
# # def preprocess(review, vocab_to_int):
# #     review = review.lower()
# #     word_list = review.split()
# #     num_list = []
# #     # list of reviews
# #     # though it contains only one review as of now
# #     reviews_int = []
# #     for word in word_list:
# #         if word in vocab_to_int.keys():
# #             num_list.append(vocab_to_int[word])
# #     reviews_int.append(num_list)
# #     return reviews_int

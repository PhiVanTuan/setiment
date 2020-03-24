import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNN_LSTM(torch.nn.Module):
    def __init__(self, weights, num_classes,num_filters,embedding_dim,hidden_dim,n_layers,window_sizes=(3, 4, 5)):
        super(CNN_LSTM, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, [window_size, embedding_dim], padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])

        self.embedding = nn.Embedding.from_pretrained(weights)
        self.lstm = nn.LSTM(len(window_sizes), hidden_dim,n_layers, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)
        self.soft_max=nn.Softmax()
    def forward(self, x,hidden):
        x = self.embedding(x)  # [B, T, E]

        # Apply a convolution + max pool layer for each window size
        x = torch.unsqueeze(x, 1)  # [B, C, T, E] Add a channel dim.
        xs = []
        for conv in self.convs:
            x2 = conv(x)
            x2 = F.relu(x2)  # [B, F, T, 1]
            x2 = torch.squeeze(x2, -1)  # [B, F, T]
            x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2)

        lstm_out,hidden=self.lstm(x,hidden)
        fc_output=self.fc(lstm_out[:,-1,:])
        out = self.soft_max(fc_output)
        return out,hidden

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



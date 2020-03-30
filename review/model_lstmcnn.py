import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_CNNTextClassifier(nn.Module):
    def __init__(self, weights, num_classes, num_filters, embedding_dim, hidden_dim, n_layers, window_sizes=(3, 4, 5)):

        super(LSTM_CNNTextClassifier, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(weights)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=0.5, batch_first=True)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, [window_size, hidden_dim], padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])

        self.fc = nn.Linear(num_filters * len(window_sizes), num_classes)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, hidden):

        x = self.embedding(x)  # [B, T, E]
        lstm_out, hidden = self.lstm(x, hidden)
        # Apply a convolution + max pool layer for each window size
        x = torch.unsqueeze(lstm_out, 1)  # [B, C, T, E] Add a channel dim.
        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(x))  # [B, F, T, 1]
            x2 = torch.squeeze(x2, -1)  # [B, F, T]
            x2 = F.max_pool1d(x2, x2.size(2))
            x2 = self.dropout(x2)  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2)
        x=self.dropout(x)
        # [B, T, E]
        x = x.view(x.size(0), -1)  # [B, F * window]
        logits = self.fc(x)  # [B, class]

        # Prediction
        probs = F.softmax(logits)  # [B, class]

        return probs, hidden

    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data
        train_on_gpu = torch.cuda.is_available()
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden
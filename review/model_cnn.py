import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnTextClassifier(nn.Module):
    def __init__(self, weights, num_classes,num_filters,drop_out=0.5,window_sizes=(3, 4, 5)):

        super(CnnTextClassifier, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(weights)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, [window_size, 300], padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])
        self.drop_out=nn.Dropout(drop_out)
        self.fc = nn.Linear(num_filters * len(window_sizes), num_classes)

    def forward(self, x,hidden):
        x = self.embedding(x)           # [batch_size, T, embeding_dim]

        # Apply a convolution + max pool layer for each window size
        x = torch.unsqueeze(x, 1)       # [batch_size, C, T, embeding_dim] Add a channel dim.
        xs = []
        for conv in self.convs:
            x2=conv(x)
            x2 = F.relu(x2)        # [batch_size, F, T, 1]
            x2 = torch.squeeze(x2, -1)  # [batch_size, F, T]
            x2 = F.max_pool1d(x2, x2.size(2))  # [batch_size, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2)            # [batch_size, F, window]
        x=self.drop_out(x)
        # FC
        x = x.view(x.size(0), -1)       # [batch_size, F * window]
        logits = self.fc(x)             # [batch_size, class]

        # Prediction
        probs = F.softmax(logits)       # [batch_size, class]


        return probs, hidden

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
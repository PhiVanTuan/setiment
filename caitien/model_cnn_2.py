import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnTextClassifier2(nn.Module):
    def __init__(self, weights, num_classes,num_filters,drop_out=0.5,window_sizes=(3, 4, 5)):

        super(CnnTextClassifier2, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(weights)
        self.n_layers=2
        self.hidden_dim=256

        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, [window_size, 300], padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])
        self.drop_out=nn.Dropout(drop_out)
        self.fc = nn.Linear(num_filters * len(window_sizes), num_classes)
        self.cosine = nn.CosineSimilarity(dim=0)
        self.fc_out=nn.Linear(num_classes,2)
        self.soft_max = nn.Softmax()

    def forward(self, x):
        batch_size=x.size(0)
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
        out_put=self.fc_out(logits)
        out_put=self.soft_max(out_put)
        # logits=logits.double()
        # list_out_test=[logits[x,:,] for x in range(batch_size)]
        #
        #
        # array=[]
        #
        # for index,v in enumerate(list_out_test):
        #     x=list_out_test[index]
        #     for i,value in enumerate(list_out_test):
        #         y = self.cosine(x, value)
        #         array.append(y)
        #
        # test = torch.stack(array,0)
        # test=test.view(batch_size,batch_size)

        # Prediction
        # probs = F.softmax(logits)       # [batch_size, class]


        return out_put

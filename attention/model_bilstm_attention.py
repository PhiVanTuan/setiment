import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable



class LSTMAttention(torch.nn.Module):
    def __init__(self, opt):

        super(LSTMAttention, self).__init__()
        self.hidden_dim = opt.hidden_size
        self.batch_size = opt.batch_size
        self.use_gpu = torch.cuda.is_available()

        self.word_embeddings = nn.Embedding.from_pretrained(opt.weights)
        self.num_layers = opt.n_layer
        # self.bidirectional = True
        self.dropout = opt.dropout
        self.bilstm = nn.LSTM(opt.embedding_dim, opt.hidden_size , batch_first=True, num_layers=self.num_layers,
                              dropout=self.dropout, bidirectional=True)
        self.hidden2label = nn.Linear(opt.hidden_size*2, opt.num_classes)
        # self.hidden = self.init_hidden()

        self.attn_fc = torch.nn.Linear(opt.embedding_dim, 1)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        train_on_gpu = torch.cuda.is_available()
        if (train_on_gpu):
            hidden = (weight.new(self.num_layers*2, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.num_layers*2, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.num_layers*2, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.num_layers*2, batch_size, self.hidden_dim).zero_())
        h = tuple([each.data for each in hidden])
        return h

    def attention(self, rnn_out, state):
        # x=[s for s in state]
        # merged_state = torch.cat(x, 1)
        merged_state_sq=state.squeeze(0) #squeeze remove tensor tai index co dimension_size=1
        merged_state_un = merged_state_sq.unsqueeze(2)
        # (batch, seq_len, cell_size) * (batch, cell_size, 1) = (batch, seq_len, 1)
        weights = torch.bmm(rnn_out, merged_state_un)
        weights_sq=weights.squeeze(2)
        weights = torch.nn.functional.softmax(weights_sq).unsqueeze(2)
        # (batch, cell_size, seq_len) * (batch, seq_len, 1) = (batch, cell_size, 1)
        x=torch.transpose(rnn_out, 1, 2)
        return torch.bmm(x, weights).squeeze(2)

    # end method attention

    def forward(self, X):
        batch_size=X.size()[0]
        embedded = self.word_embeddings(X)
        # packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        hidden = self.init_hidden(batch_size)  #
        rnn_out, hidden = self.bilstm(embedded, hidden)
        h_n, c_n = hidden
        h_n_final_layer = torch.cat((h_n[-2, :, :], h_n[-1, :, :]),1)

        # h_n_final_layer = h_n.view(self.num_layers,2,batch_size,self.hidden_dim)[-1,:,:,:]
        # h_n (num_direction*num_layer,batch_size,hidden_dim)
        attn_out = self.attention(rnn_out, h_n_final_layer)
        logits = self.hidden2label(attn_out)
        out_put=torch.nn.functional.softmax(logits)
        return logits
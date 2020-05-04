import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable



class LSTMAttention2(torch.nn.Module):
    def __init__(self, opt):

        super(LSTMAttention2, self).__init__()
        self.hidden_dim = opt.hidden_size
        self.batch_size = opt.batch_size
        self.use_gpu = torch.cuda.is_available()
        self.opt=opt
        self.word_embeddings = nn.Embedding.from_pretrained(opt.weights)
        self.num_layers = opt.n_layer
        # self.bidirectional = True
        self.dropout = opt.dropout
        self.bilstm = nn.LSTM(opt.embedding_dim, opt.hidden_size , batch_first=True, num_layers=self.num_layers,
                              dropout=self.dropout, bidirectional=True)
        self.hidden2label = nn.Linear(opt.hidden_size*2, opt.num_classes)
        # self.hidden = self.init_hidden()

        self.attn_fc = torch.nn.Linear(opt.embedding_dim, 1)
        self.out_fc=nn.Linear(opt.num_classes,5)
        self.soft_max=nn.Softmax()
        self.cosine=nn.CosineSimilarity(dim=0)

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
        out_put=self.out_fc(logits)
        out_put=self.soft_max(out_put)
        # logits=logits.double()
        # list_out_test=[logits[x,:,] for x in range(batch_size)]
        #
        # x=list_out_test[0]
        # y=list_out_test[0]
        # array=[]
        #
        # for index,v in enumerate(list_out_test):
        #     x=list_out_test[index]
        #     for i,value in enumerate(list_out_test):
        #         y = self.cosine(x, value)
        #         array.append(y)
        #
        # test = torch.stack(array,0)
        # test=test.view(40,40)

        return out_put
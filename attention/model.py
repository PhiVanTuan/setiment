import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


class AttentionModel(torch.nn.Module):
    def __init__(self, opt):
        super(AttentionModel, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

        --------

        """

        self.n_layer=opt.n_layer
        self.output_size = opt.num_classes
        self.hidden_size = opt.hidden_size
        self.embedding_length = opt.embedding_dim
        self.word_embeddings = nn.Embedding.from_pretrained(opt.weights)
        self.lstm = nn.LSTM(opt.embedding_dim, opt.hidden_size, opt.n_layer,dropout=0.5, )
        self.label = nn.Linear(opt.hidden_size, opt.num_classes)

    # self.attn_fc_layer = nn.Linear()

    def attention_net(self, lstm_output, final_state):
        """
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.

        Arguments
        ---------

        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM

        ---------

        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.

        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)

        """

        hidden = final_state.squeeze(0) # hidden.size() = (batch_size, hidden_size)
        hidden=hidden.unsqueeze(2)
        bmm_weight=torch.bmm(lstm_output, hidden)
        attn_weights = bmm_weight.squeeze(2) # attn_weights.size() = (batch_size, num_seq)
        soft_attn_weights = F.softmax(attn_weights, 1) #soft_attn_weights.size() = (batch_size, num_seq)
        lstm_transpose=lstm_output.transpose(1, 2)
        soft_attn_weights_uq=soft_attn_weights.unsqueeze(2)
        new_hidden_state = torch.bmm(lstm_transpose, soft_attn_weights_uq)
        result_hidden=new_hidden_state.squeeze(2)#new_hidden_state.size() = (batch_size, hidden_size)
        return result_hidden

    def forward(self, input_sentences):
        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class which receives its input as the new_hidden_state which is basically the output of the Attention network.
        final_output.shape = (batch_size, output_size)

        """
        batch_size = input_sentences.size(0)
        input = self.word_embeddings(input_sentences)
        input = input.permute(1, 0, 2)
        train_on_gpu = torch.cuda.is_available()
        if train_on_gpu:
           h_0 = Variable(torch.zeros(self.n_layer, batch_size, self.hidden_size).cuda())
           c_0 = Variable(torch.zeros(self.n_layer, batch_size, self.hidden_size).cuda())
        else:
            h_0 = Variable(torch.zeros(self.n_layer, batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(self.n_layer, batch_size, self.hidden_size))
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        # final_hidden_state.size() = (1, batch_size, hidden_size)
        output = output.permute(1, 0, 2)
        # output.size() = (batch_size, num_seq, hidden_size)

        attn_output = self.attention_net(output, final_hidden_state)
        logits = self.label(attn_output)
        out=F.softmax(logits)
        return out

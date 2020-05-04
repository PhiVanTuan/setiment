import torch
import numpy as np
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, opt):

        super(Model, self).__init__()

        self.batch_size = opt.batch_size
        self.use_gpu = torch.cuda.is_available()
        self.opt=opt

        # self.hidden = self.init_hidden()

        self.fc = nn.Linear(opt.num_classes, 3)
        self.soft_max=nn.Softmax()

    # end method attention

    def forward(self, X):
        out_put=self.fc(X)
        out_put=self.soft_max(out_put)
        return out_put
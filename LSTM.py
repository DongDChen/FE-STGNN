import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from torch.autograd import Variable
import os

class LSTMCell(nn.Module):
    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory' cell.
    http://www.bioinf.jku.at/publications/older/2604.pdf

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.EC = nn.Linear(input_size, hidden_size, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden, EC_in):
        hx, cx = hidden

        x = x.view(-1, x.size(1))

        EC_in = EC_in.view(-1, x.size(1))

        EC_in = self.EC(EC_in)

        gates = self.x2h(x) + self.h2h(hx)

        gates = gates.squeeze()

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        # print("ingate size", ingate.size())
        # print("forgetgate size", forgetgate.size())
        # print("cellgate size", cellgate.size())
        # print("outgate size", outgate.size())
        # print("EC_in size", EC_in.size())

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        # forgetgate = torch.sigmoid(forgetgate*EC_in)    # FC+EC
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)

        hy = torch.mul(outgate, torch.tanh(cy))

        return (hy, cy)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.lstm = LSTMCell(input_dim, hidden_dim, layer_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)



    def forward(self, x, EC_in):


        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(device)
        c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(device)
        outs = []

        print("FC_in.size",x.size())
        print("EC_in.size", EC_in.size())
        cn = c0[0, :, :]
        hn = h0[0, :, :]

        for seq in range(x.size(1)):
            hn, cn = self.lstm(x[:, seq, :], (hn, cn), EC_in[:, seq, :])
            outs.append(hn)

        last_hidden_out = outs[-1].squeeze()

        out = self.fc(last_hidden_out)
        # out.size() --> 100, 10
        return last_hidden_out, out
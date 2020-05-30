import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

max_len = 512
embed_size = 300
num_class = 8
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLP(nn.Module):
    '''
    Multiayers Perceptron
    '''

    def __init__(self, params):
        super().__init__()
        hidden_size = params['hidden_size']
        dropout = params['dropout']
        self.fc = nn.Sequential(
            nn.Linear(max_len * embed_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_class),
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class DAN(nn.Module):
    '''
    Deep Average Network
    '''

    def __init__(self, params):
        super().__init__()
        word_dropout = params['word_dropout']
        dropout = params['dropout']
        hidden_size = params['hidden_size']
        self.word_dropout = word_dropout
        self.fc = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_class)
        )

    def forward(self, x):
        # x.size: [64, 512, 300]
        batch = x.shape[0]
        # word dropout
        if(self.training):
            mask = torch.bernoulli(torch.ones(
                batch, max_len) * (1 - self.word_dropout)).to(device)
        else:
            mask = torch.ones(batch, max_len).to(device)

        x[mask == 0] *= 0
        word_sum = torch.sum(mask == 1, dim=1).reshape((batch, 1))
        x = torch.sum(x, dim=1) / word_sum
        x = self.fc(x)
        return x


class CNN(nn.Module):
    '''
    Convolutional Neural Network
    '''

    def __init__(self, params):
        super().__init__()
        num_filters = params['num_filters']
        filter_size = params['filter_size']
        dropout = params['dropout']

        self.convs = nn.ModuleList([nn.Conv2d
                                    (1, num_filters, (k, embed_size)) for k in filter_size])
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_filters * len(filter_size), num_class)
        )

    def conv_pool(self, x, conv):
        # x: (b, max_len, embed_size)
        # conv(x): (b, num_filters, max_len , 1)
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.shape[2]).squeeze(2)
        return x

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.cat([self.conv_pool(x, conv) for conv in self.convs], 1)
        x = self.fc(x)
        return x


class RNN(nn.Module):
    '''
    BiLSTM with Self Attention
    '''

    def __init__(self, params):
        super().__init__()
        hidden_size = params['hidden_size']
        self.Attention = params['Attention']
        # input_size, hidden_size
        # num_layers default = 1
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            bidirectional=True, batch_first=True)
        if(self.Attention):
            self.w = nn.Parameter(torch.zeros(hidden_size * 2))

        self.fc = nn.Linear(hidden_size * 2, num_class)

    def forward(self, x):
        # lstm(x) = output, (h_n, c_n)
        # h_0, c_0 default zero
        H, _ = self.lstm(x)
        if(self.Attention):
            M = torch.tanh(H)
            alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
            out = H * alpha
            out = torch.sum(out, dim=1)
            out = self.fc(out)
        else:
            out = self.fc(H[:, -1, :])
        return out


class RCNN(nn.Module):
    '''
    Recurrent Convolutional Neural Network
    '''

    def __init__(self, params):
        super().__init__()
        hidden_size = params['hidden_size']
        num_filters = params['num_filters']
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            bidirectional=True, batch_first=True)
        self.conv = nn.Conv2d(
            1, num_filters, (1, 2 * hidden_size + embed_size))
        self.maxpool = nn.MaxPool1d(max_len)
        self.fc = nn.Linear(num_filters, num_class)

    def forward(self, x):
        H, _ = self.lstm(x)
        x = torch.cat((x, H), 2).unsqueeze(1)
        # [c_l; e; e_r]
        x = self.conv(x).squeeze(3)
        # x = torch.tanh(x)
        x = F.relu(x)
        x = F.max_pool1d(x, x.shape[2]).squeeze(2)
        x = self.fc(x)
        return x

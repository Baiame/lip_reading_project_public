import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

class Network(nn.Module):
    def __init__(self, img_width = 300, img_height = 150):
        super(Network, self).__init__()

        self.img_width = img_width
        self.img_height = img_height

        self.dropout = nn.Dropout(0.2)
        
        self.lstm_layers = nn.LSTM(input_size = 40, hidden_size = 32, num_layers = 2, batch_first=True)

        self.linear_layer1 = nn.Linear(in_features = 32, out_features = 16)
        torch.nn.init.xavier_uniform(self.linear_layer1.weight)

        self.linear_layer2 = nn.Linear(in_features = 16, out_features = 4)
        torch.nn.init.xavier_uniform(self.linear_layer2.weight)

        #self.linear_layer3 = nn.Linear(in_features = 16, out_features = 4)
        #torch.nn.init.xavier_uniform(self.linear_layer2.weight)

    def forward(self, input):
        """
        The input is a batch of sequences of mouth shapes
        """

        # LSTM layer
        input = self.dropout(input)
        lstm_out, (h_t, c_t) = self.lstm_layers(input, None)
        output = h_t[-1]
        #output = output[:, -1, :]

        # linear layers to output an array of 4 elements
        output = F.relu(self.linear_layer1(output))
        output = self.linear_layer2(output)
        #output = self.linear_layer3(output)
 
        # softmax
        output = torch.nn.functional.softmax(output)

        # return the list as a stacked tensor
        return output
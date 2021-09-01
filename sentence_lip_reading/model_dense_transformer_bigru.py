import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder
from dense_modules import _DenseLayer, _DenseBlock, _Transition



###################################################################
######################### Transformer Class #######################
###################################################################

class TransformerModel(nn.Module):

    def __init__(self, input_size, embedding_size, number_heads, number_hiddens, number_layers):
        """
        input_size: Input data size
        embedding_size: Embedding size for the current input
        number_heads: Number of heads for the encoder
        number_hiddens: The number of hidden units in the encoder
        number_layers: The number of layers in the encoder
        """
        super(TransformerModel, self).__init__()
        

        # Initialize the FC layer, and the position encoding
        self.linear = nn.Linear(input_size, embedding_size)
        self.pos_encoding = PositionalEncoding(embedding_size)

        # Initialize the encoder with n layers, that each has heads and hidden units.
        encoder_layers = TransformerEncoderLayer(embedding_size, number_heads, number_hiddens, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, number_layers)


    def forward(self, input_seq):
        """
        Input:
        input_seq: a tensor of shape (sequence_len, batch_size)

        Output:
            output: a tensor of shape (sequence_len, batch_size, vocabulary_size)
        """
        # Embed the source sequences and add the positional encoding.
        input_seq = self.linear(input_seq)
        input_seq = self.pos_encoding(input_seq)
        output = self.transformer_encoder(input_seq)     

        return output


class PositionalEncoding(nn.Module):
    """
    Modify the input with positional encoding, depending on the time
    Implementation from pytorch documentation
    """

    def __init__(self, model_d, length_max=5000):
        super(PositionalEncoding, self).__init__()


        log_value = 10000.0

        # Initialize the encoding
        pos_enco = torch.zeros(length_max, model_d)

        # Initialise the positions
        position = torch.arange(0, length_max, dtype=torch.float).unsqueeze(1)

        division = (torch.arange(0, model_d, 2).float()) * (-math.log(log_value) / model_d)
        division = torch.exp(division)

        # Encoding using cos and sin
        pos_enco[:, 0::2] = torch.sin(position * division)
        pos_enco[:, 1::2] = torch.cos(position * division)

        # Return the encoding
        pos_enco = pos_enco.unsqueeze(0).transpose(0, 1)

        # Register the buffer
        self.register_buffer('pos_enco', pos_enco)

    def forward(self, x):
        """
        x: type = tensor, shape = (sequence_len, batch, embed_size)

        Output a tensor of shape (sequence_len, batch, embed_size)
        """
        x += self.pos_enco[:x.size(0), :]

        return x



###################################################################
########################### Dense3D Class #########################
###################################################################




class Dense3D(torch.nn.Module):
    """
    Create the front_end network
    Implementation of a densely connected convolutionnal network
    """
    def __init__(self, growth=8, number_initial_features=32, batch_norm_size=4, dropout=0):
        super(Dense3D, self).__init__()

        dense3D = True
        config = 1

        if config == 1:
            # 2 blocks of 4 layers
            configuration = (4, 4)
        else:
            # 3 blocks of 4 layers
            configuration = (4, 4, 4)

        kernel = (1, 2, 2)
        stride = (1, 2, 2)

        # Initialise the network with a single layer
        first_layer = OrderedDict([
            ('conv0', nn.Conv3d(3, number_initial_features, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))),
            ('norm0', nn.BatchNorm3d(number_initial_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=kernel, stride=stride)),
        ])
        self.features = nn.Sequential(first_layer)


        number_features = number_initial_features

        # Add 2 blocks of 4 layers
        k = 0
        for number_layers in configuration:

            if dense3D == True:
                # Add a block
                block = _DenseBlock(number_layers=number_layers, number_features=number_features,
                                    batch_norm_size=batch_norm_size, growth=growth, dropout=dropout)

                name = 'denseblock%d' % (k + 1)
                self.features.add_module(name, block)

                number_features = number_features + number_layers * growth

                if k != len(configuration) - 1:
                    trans = _Transition(number_features_in=number_features, number_features_out=number_features)
                    name = 'transition%d' % (k + 1)
                    self.features.add_module(name, trans)
            else:
                # Possibility to change the architecture
                pass

            k+=1

        batch_norm = nn.BatchNorm3d(number_features)
        avg_pooling = nn.AvgPool3d(kernel_size=kernel, stride=stride)
        self.features.add_module('norm%d' % (len(configuration)), batch_norm)
        self.features.add_module('pool', avg_pooling)

    def forward(self, x):
        return self.features(x)



#################################################################################################################
################################################## The Model ####################################################
#################################################################################################################

class LipReadingNetwork(torch.nn.Module):
    def __init__(self, transformer=True, lstm=False, bidir=True):
        super(LipReadingNetwork, self).__init__()

        # If true, the model has a transformer back-end, else LSTM or GRU
        self.transformer = transformer
        self.lstm = lstm
        self.bidir = bidir

        # Dense3D Front End Network
        self.Dense3D = Dense3D()

        if transformer:
            # Transformer Back End Network
            self.transformer_encoder = TransformerModel(3072, 512, 8, 512, 3)

        elif lstm == True:
            self.lstm_layer1 = nn.LSTM(3072, 256, 1, bidirectional=self.bidir)
            self.lstm_layer2 = nn.LSTM(512, 256, 1, bidirectional=self.bidir)
            self.dropout = nn.Dropout(0.5)

        else:
            self.gru_layer1 = nn.GRU(3072, 256, 1, bidirectional=self.bidir)
            self.gru_layer2 = nn.GRU(512, 256, 1, bidirectional=self.bidir)
            self.dropout = nn.Dropout(0.5)

        # Output of the model
        self.fully_connected = nn.Linear(512, 28)



    def forward(self, x):
        

        # Pass the input tensor containing the frames through the 3D dense network
        # (B, C=3, T, H=64, W=128) --> (B, C=96, T, H=4, W=8)
        x = self.Dense3D(x)

        # Rearrange the input before encoder
        # (B, C, T, H, W) -> (T, B, C, H, W)
        x = x.permute(2, 0, 1, 3, 4)
        x = x.contiguous()

        # Concatenate Channel/Height/Width (B, C, T, H, W) -> (T, B, C*H*W)
        B = x.size(0)   # batch size
        C = x.size(1)   # channel size
        x = x.view(B, C, -1)

        # Transformer
        # (T, B, Emb=3072) --> (T, B, Emb=512)
        if self.transformer:
            x = self.transformer_encoder(x)

        # LSTM or Bi-LSTM
        elif self.lstm:
            x, _, _ = self.lstm_layer1(x)
            x = self.dropout(x)
            x, _, _ = self.lstm_layer2(x)
            x = self.dropout(x)

        # GRU or Bi-GRU
        else:
            x, _ = self.gru_layer1(x)
            x = self.dropout(x)
            x, _ = self.gru_layer2(x)
            x = self.dropout(x)


        # Fully connected layer
        #(T, B, Emb=512) --> (T, B, Emb=28)
        x = self.fully_connected(x)


        # (T, B, Emb) -> (B, T, Emb)
        x = x.permute(1, 0, 2)
        x = x.contiguous()

        return x
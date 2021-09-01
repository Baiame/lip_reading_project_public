import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models


import numpy as np

class Net(nn.Module):
    def __init__(self, img_width = 300, img_height = 150):
        super(Net, self).__init__()

        self.img_width = img_width
        self.img_height = img_height

        self.cnn_layers = nn.Sequential(
            # 300 --> 150 and 150 --> 75
            nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = (4, 4), stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = 8),
            #MaxPool2d(kernel_size = 2, stride = 2),

            # 150 --> 50 and 75 --> 38
            nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = (7, 3), stride = (3, 2), padding = (2, 1)),
            nn.BatchNorm2d(num_features = 16),
            #MaxPool2d(kernel_size = 2, stride = 2),

            # 50 --> 25 and 38 --> 21
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (4, 4), stride = (2, 2), padding = (1, 3)),
            nn.BatchNorm2d(num_features = 32),
            #MaxPool2d(kernel_size = 2, stride = 2),

            # 25 --> 13 and 21 --> 13
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (5, 3), stride = (2, 2), padding = (2, 3)),
            nn.BatchNorm2d(num_features = 64),
            #MaxPool2d(kernel_size = 2, stride = 2),

            # 13 --> 4 and 13 --> 4
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 3, padding = 0),
            nn.BatchNorm2d(num_features = 128),
            #MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 1, padding = 0)
            #MaxPool2d(kernel_size = 2, stride = 2),
        )

        self.linear_layer1 = nn.Linear(in_features = 256, out_features = 128)

        self.lstm_layers = nn.LSTM(input_size = 1, hidden_size = 64, num_layers = 2, batch_first=True)

        self.linear_layer2 = nn.Linear(in_features = 64, out_features = 4)

    def forward(self, input):

        # the input is a batch videos
        batch_tensor = []

        for video in input:
            video_tensor = []
            # [nb_images, weight, width, channel] --> [nb_images, channel, width, weight]
            video = video.view(video.shape[0], 3, 300, 150).cuda()
            
            i = 0
            for frame in video:
                frame = frame.view(1, 3, 300, 150)

                # convolution on each frame
                conv_frame = self.cnn_layers(frame)
                conv_frame = conv_frame.view(1, 256)

                # dense layer on each frame
                conv_frame = self.linear_layer1(conv_frame)
                conv_frame = conv_frame.view(128)
                
                # concatenate the convolved frames
                if i == 0:
                    video_tensor = conv_frame
                else:
                    video_tensor = torch.cat((video_tensor, conv_frame), 0)
                i += 1

            #video_tensor = torch.Tensor(video_tensor)
            #video_tensor = video_tensor.view(1, len(video_tensor, 1))

            # pass the frame through LSTM layer
            video_tensor = video_tensor.view(1, video_tensor.shape[0], 1)
            video_tensor, (h_n, c_n) = self.lstm_layers(video_tensor)

            # only get the last element
            video_tensor = video_tensor[-1][-1]

            # linear layer to output an array of 4 elements
            video_tensor = self.linear_layer2(video_tensor)

            # softmax
            video_tensor = torch.nn.functional.softmax(video_tensor)

            # append the prediction of the video to the tensor
            batch_tensor.append(video_tensor)

        # return the list as a stacked tensor
        return torch.stack(batch_tensor, dim = 0)


class Net_v2(nn.Module):
    def __init__(self, img_width = 224, img_height = 224):
        super(Net_v2, self).__init__()
        self.img_width = img_width
        self.img_height = img_height

        # Layers
        # import already trained mobilenet and disallow training
        self.mobilenet_v2 = models.mobilenet_v2(pretrained=True)
        for param in self.mobilenet_v2.parameters():
            param.requires_grad = False

        # fine tuning the last layer of mobile net
        #self.number_features = self.mobilenet_v2.classifier[1].in_features
        #self.mobilenet_v2.classifier[1] = nn.Linear(self.number_features, 1000)

        self.lstm_layer = nn.LSTM(input_size = 1000, hidden_size = 2048, num_layers = 2, batch_first=True, bidirectional=False, dropout=0.2)
        self.linear_layer = nn.Linear(in_features = 2048, out_features = 512)
        self.linear_layer2 = nn.Linear(in_features = 512, out_features = 4)


    def forward(self, input):
        """
        Input must be a batch of padded sequences: Batch*Sequence*Height*Width*Channels
        """
        batch_tensor = []
        
        # use mobilenet convolution on each image
        for video in input:
            # S*H*W*C --> S*C*H*W
            video = video.permute(0, 3, 1, 2)
            sequence = self.mobilenet_v2(video)
            batch_tensor.append(sequence)
        
        # stack all videos in a tensor
        output = torch.stack(batch_tensor, dim = 0)
        # LSTM layer
        _, hn = self.GRU_layer(output)
        output = hn[-1]
        # Dense layers
        output = self.linear_layer(output)
        output = self.linear_layer2(output)
        # Softmax activation function
        output = torch.nn.functional.softmax(output)

        return output

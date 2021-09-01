# import the necessary packages
from imutils import video
from imutils.video import VideoStream
from imutils import face_utils
from model_one_word import Net_v2
from dataset_one_word import VideoDataset
from train_one_word import my_collate_fn
from torch.nn.utils.rnn import pad_sequence

import datetime
import argparse
import imutils
import time
import dlib
import cv2
import os
import numpy as np
import torch
from torchvision import transforms


import torchvision.models as models

model = Net_v2()
model.load_state_dict(torch.load('./word_lip_reading/models/model_5_speakers_high_res.pt'))

model.cuda()


def extract_tensor_from_folder(path):
    """
    Extract a video tensor from a folder that contains the images of the video
    """
    image_list = os.listdir(path)
    video_tensor = []
    for image in image_list:
        image_path = path + image
                
        # load each image in a list only if it is a .jpg
        extension = os.path.splitext(image_path)[1]
        if extension == '.jpg':
            frame_np = cv2.imread(image_path)
            video_tensor.append(frame_np)
    return video_tensor


if __name__ == '__main__':
    video_tensor = extract_tensor_from_folder(path='./lombardgrid_2_high_res/one_word_dataset/valid/place/pwbj9a/')
    #video_tensor = extract_tensor_from_folder(path='./tests/myriam_place/')
    test_dataset = VideoDataset(root_dir='./lombardgrid_2_high_res/one_word_dataset/', split='train_test')
    test_loader = torch.utils.data.DataLoader(
                                            dataset=test_dataset, 
                                            batch_size=64, 
                                            shuffle=True, 
                                            collate_fn=my_collate_fn)

    correct = 0
    total = 0

    with torch.no_grad():
        for i, (videos, labels) in enumerate(test_loader):
            videos = pad_sequence(sequences=videos, batch_first=True)
            outputs = model(videos)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            total += labels.size(0)
            if torch.cuda.is_available():
                correct += (predicted.cpu() == labels.cpu()).sum()
            else:
                correct += (predicted == labels).sum()

        accuracy = 100 * correct // total
        print('Test Accuracy : {}%'.format(accuracy))


    model.eval()

    with torch.no_grad():
        for i, (videos, labels) in enumerate(test_loader):
            videos = pad_sequence(sequences=videos, batch_first=True)
            outputs = model(videos)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            total += labels.size(0)
            if torch.cuda.is_available():
                correct += (predicted.cpu() == labels.cpu()).sum()
            else:
                correct += (predicted == labels).sum()

        accuracy = 100 * correct // total
        print('Test Accuracy : {}%'.format(accuracy))   

    video_tensor = torch.FloatTensor(video_tensor).cuda()
    print(video_tensor.size())
    video_tensor = torch.unsqueeze(video_tensor, 0)
    print(video_tensor.size())

    output = model(video_tensor)
    print(output)

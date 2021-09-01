import cv2
import os
from data_augmentation import *
import torch
import string
import editdistance
import numpy as np
from torch.utils.data import Dataset
from array_text_conversion import *


class LipsDataset(Dataset):     # inherit Dataset

    def __init__(self, alignment_path, file_list, video_padding, text_padding, split, root='./'):

        # Load attributes
        self.text_padding = text_padding
        self.alignment_path = alignment_path
        self.split = split
        self.video_padding = video_padding

        # Open the files
        try:
            file = open(file_list, 'r')
            self.videos = [path_line.strip() for path_line in file.readlines()]
        finally:
            file.close()

        # store the paths, the speakers and the name
        self.video_speaker_name = []
        for video in self.videos:
            items = video.split('/')
            # video is the path to the folder that contains the images of the video, items[-5] is the speaker (ex: s23), items[-2] is the name of the folder (ex: bram4p)
            self.video_speaker_name.append((video, items[-5], items[-2]))

    def __len__(self):
        """Return data length"""
        return len(self.video_speaker_name)

    def __getitem__(self, index):
        """
        For indexing: Dataset[i] gets to i-th sample
        """
        # Retrieve data
        data = self.video_speaker_name[index]

        # Split the data into video, speaker and name
        video = data[0]
        speaker = data[1]
        name = data[2]


        # load annotations from alignment file
        extension = '.align'
        anno_path = os.path.join(self.alignment_path, speaker, 'align', name + extension)
        annotation = self.load_annotation(anno_path)
        
        # load video and associated alignement file using the path
        # vid is (T, H, W, C)
        video = self.load_video(video)

        if(self.split == 'train'):
            video = self.data_augmentation(video)

        # Normalize the color (values between 0 and 1)
        video = video / 255.0                   
        
        # get the length of the video (nb of images) and the length of the alignment file (timestamps and words)
        video_length = video.shape[0]
        annotation_length = annotation.shape[0]
        
        # pad the videos and the alignment files to the same length (to allow stacking in the same batch)
        video = self.add_padding(video, self.video_padding)
        annotation = self.add_padding(annotation, self.text_padding)

        video = video.transpose(3, 0, 1, 2)
        video = torch.FloatTensor(video)

        annotation = torch.LongTensor(annotation)
        # convert video (T, H, W, C) --> (C, T, H, W)
        return {'video': video, 
            'text': annotation,
            'text_length': annotation_length,
            'video_length': video_length}    
        


    def is_jpeg(self, filename):
        """Return true if jpeg file"""
        idx = filename.find('.jpg') != -1
        return idx != -1
    
    def sort_file(self, file):
        return int(os.path.splitext(file)[0])

    def load_video(self, path):
        """
        Load a video using the path and return an array
        """ 
        # List directory
        loaded_files = os.listdir(path)
        # Keep only jpeg
        loaded_files = list(filter(self.is_jpeg, loaded_files))
        # Sort
        loaded_files = sorted(loaded_files, key=self.sort_file)
        
        
        # Load the image and verify it's not None
        np_array = []
        for file in loaded_files:
            file_path = os.path.join(path, file)
            image = cv2.imread(file_path)
            if image is not None:
                image = cv2.resize(image, (128, 64), interpolation=cv2.INTER_LANCZOS4)
                np_array.append(image)

        np_array = np.stack(np_array, axis=0).astype(np.float32)
        return np_array


    def data_augmentation(self, video):
        """ Apply data augmentation techniques to the input video"""
        # For this project: only horizontal flip
        #video = randomCropping(video)
        return HorizontalFlip(video)


    def load_annotation(self, path):
        """ Load the alignment file"""
        # Annotations to filer
        to_filter = ['SIL', 'SP']
        # open the file
        with open(path, 'r') as file:
            # Retrieve all words
            lines = []
            for line in file.readlines():
                lines.append(line.strip().split(' '))

            # Compute sentence
            text = []
            for line in lines:
                text.append(line[2])
            text = list(filter(lambda s: not s.upper() in to_filter, text))
        
        # Concatenate all words to a string
        text = ' '.join(text).upper()
        # Convert string to array
        array = convert_text_to_array(text, 1)
        return array


    def print_annotation(self, path):
        """ Print annotation file under path"""
        with open(path, 'r') as file:
            for line in file.readlines():
                print(line)


    def add_padding(self, array, length):
        """ Apply padding to the input array to equal the input length"""
        array = [array[_] for _ in range(array.shape[0])]
        size_to_add = array[0].shape
        nb_iter = length - len(array)
        for _ in range(nb_iter):
            array.append(np.zeros(size_to_add))
        return np.stack(array, axis=0)





#################################################################################################################
#################################################################################################################
#################################################################################################################

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    train_dataset = LipsDataset('C:/Users/aurel/Downloads/grid_dataset_downloaded/GRID_align_txt',  'C:/Users/aurel/Downloads/grid_dataset_downloaded/paths_list_250_by_spk.txt', 75, 100, 'train')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    for (i_iter, input) in enumerate(train_loader):
        if i_iter == 0:
            print(input)

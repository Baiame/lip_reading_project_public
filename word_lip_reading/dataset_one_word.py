import torch
import os
import cv2
from torch.utils.data import Dataset, DataLoader

cuda = True if torch.cuda.is_available() else False


class VideoDataset(Dataset):
    """ Dataset of one word videos from the GridCorpus Dataset"""

    def build_list(self, directory):
        """
        Build a list that contain the paths to folders (each folder contains the images of one video)
        
        """
        list_of_data = []
        # parse the labels
        for i, label in enumerate(self.label_list):
            dir = self.root_dir + self.split + '/' + label +'/'
            print(i, dir)
            folders = os.listdir(dir)
            # for each label, parse all the videos associated to it
            for folder in folders:
                folder_path = dir + folder + '/'
                list_of_data.append((i, folder_path))
        return list_of_data


    def __init__(self, root_dir, split, transform = None):
        """
        Args:
            root_dir (string): Directory with all the data files.
            split    (string): In ["train", "valid", "test"].
        """
        self.root_dir = root_dir
        self.split = split
        self.label_list = ['bin', 'lay', 'place', 'set']
        self.pathList = self.build_list(self.root_dir)
        self.transform = transform


    def __len__(self):
        # size of the dataset
        return len(self.pathList)


    def __getitem__(self, idx):
        """
        Load the video of the dataset associated to the index in a tensor
        """
        label, path = self.pathList[idx]
        image_list = os.listdir(path)
        video_tensor = []
        # parse the images in the folder
        for image in image_list:
            image_path = path + image
            
            # load each image in a list only if it is a .jpg
            extension = os.path.splitext(image_path)[1]
            if extension == '.jpg':
                frame_np = cv2.imread(image_path)
                video_tensor.append(frame_np)

        # convert list to tensor
        video_tensor = torch.FloatTensor(video_tensor).cuda()
        # return the sample
        # sample = {
        #     'video' : video_tensor,
        #     'label' : torch.LongTensor([label])
        # }

        #if self.transform:
            #video_tensor = self.transform(frame_np)


        return video_tensor, label


#################################################################################################
#################################################################################################
#################################################################################################
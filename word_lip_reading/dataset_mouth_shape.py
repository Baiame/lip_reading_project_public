import torch
import os
import os.path

import cv2
from torch.utils.data import Dataset, DataLoader

cuda = True if torch.cuda.is_available() else False

class ShapeDataset(Dataset):
    """ Dataset of mouth shapes of one word videos """

    def build_list(self, directory):
        """
        Build a list that contain the paths to folders (each folder contains the mouths data file)
        """

        list_of_data = []
        # parse the labels
        for i, label in enumerate(self.label_list):
            dir = self.root_dir + label +'/'
            print(i, dir)
            folders = os.listdir(dir)
            # for each label, parse all the videos associated to it
            for folder in folders:
                folder_path = dir + folder + '/'
                list_of_data.append((i, folder_path))
        return list_of_data


    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.label_list = ['bin', 'lay', 'place', 'set']
        self.pathList = self.build_list(self.root_dir)


    def __len__(self):
        # size of the dataset
        return len(self.pathList)


    def __getitem__(self, idx):
        """
        Load the video of the dataset associated to the index in a tensor
        """
        label, path = self.pathList[idx]
        file_list = os.listdir(path)
        # parse the images in the folder
        for pt_file in file_list:
            file_path = path + pt_file
            extension = os.path.splitext(file_path)[1]

            # choose only the file that has the tensor weights
            if extension == '.pt':
                tensor = torch.load(file_path)              # size Nx20x2 because the shape is made of 20 (x,y) tuples
                
                # Change type Int to type Float
                tensor = tensor.type(torch.FloatTensor)     # On cpu for now

                # normalization of the coordinates
                #tensor[:,:,0] /= 300   # x
                #tensor[:,:,1] /= 150   # y

                tensor = tensor.view(tensor.size()[0], 40)  # resize to Nx40

        return tensor, label
import os
import random

DATASET_FOLDER_PATH = 'C:/Users/aurel/Downloads/grid_dataset_downloaded/'

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        #print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for d in dirs:
            print('{}'.format(d))

if __name__ == '__main__':

    list_of_data = []

    # parse the labels
    for idx_speaker in range(1, 35):
        list_of_data_spk = []
        # no speaker 21
        if (idx_speaker != 21):
            dir = DATASET_FOLDER_PATH + 'lip/s{}/'.format(str(idx_speaker)) + 'video/mpg_6000/'
            folders = os.listdir(dir)
            # for each label, parse all the videos associated to it
            for folder in folders:
                folder_paths = dir + folder + '/'
                # append only if the folder is not empty (sometimes a folder can be empty due to a bug)
                if len(os.listdir(folder_paths)) == 75:
                    list_of_data_spk.append(folder_paths)
            list_of_data.extend(random.choices(list_of_data_spk, k=250))
    print(list_of_data)

    with open('C:/Users/aurel/Desktop/Imperial/Project_Lipreading/lip_reading_project/lipreading_with_transformers/paths_list.txt', 'w') as f:
        f.write("\n".join(list_of_data))
import os
import json
import cv2
from video_processing import extract_mouth_video
from video_processing import save_array_of_images_as_jpg
from video_processing import save_mouth_shape_to_file
import torch

FACE_PREDICTOR_PATH = './shape_predictor_68_face_landmarks.dat'
JSON_FOLDER = './lombardgrid/json/'
ALIGN_FOLDER = './lombardgrid/alignment/'
VIDEO_FOLDER= './lombardgrid/front/'
RESULT_FOLDER = './lombardgrid/one_word_dataset/'
SPLIT = 'train/'

index_first_speaker = 14 #2
index_last_speaker = 30 #55

FPS = 24

def extract_frame_indices_first_word(video_path, align_path, name):
    # open alignment file
    with open(align_path) as json_file:
        data = json.load(json_file)[name]
        beginning_time = 0
        end_time = 0
        # loop over the timesteps
        i = 0
        for timestep in data:
            # according to the word, extract the time of beggining and end of the word
            if timestep['phone'] == 'b_B':
                beginning_time = float(timestep['offset'])
                end_time = float(data[i + 3]['offset'])         # 3 syllables later: offset of the next word
                label_of_word = 'bin'
                break
            elif timestep['phone'] == 'l_B':
                beginning_time = float(timestep['offset'])
                end_time = float(data[i + 2]['offset'])         # 2 syllables later
                label_of_word = 'lay'
                break
            elif timestep['phone'] == 'p_B':
                beginning_time = float(timestep['offset'])
                end_time = float(data[i + 4]['offset'])         # 4 syllables
                label_of_word = 'place'
                break
            elif timestep['phone'] == 's_B':
                beginning_time = float(timestep['offset'])
                end_time = float(data[i + 3]['offset'])         # 3 syllables
                label_of_word = 'set'
                break
            i += 1
        
        # return the indices of the first and last frames of the word ,+3

        return (int(beginning_time * FPS) - 1, int(end_time * FPS) + 1, label_of_word)



if __name__ == "__main__":
    # video_path = './lombardgrid/front/s4_l_pbik5n.mov'
    # align_path = './lombardgrid/alignment/s4_l_pbik5n.json'
    # name = 's4_l_pbik5n'
    # idx_beg, idx_end, label_of_word = extract_frame_indices_first_word(video_path, align_path, name)
    # print(idx_beg)
    # print(idx_end)
    # print(label_of_word)
    # mouth_frames = extract_mouth_video(video_path, FACE_PREDICTOR_PATH)
    # print(len(mouth_frames))
    # mouth_frames = mouth_frames[idx_beg:idx_end]
    # print(len(mouth_frames))
    # store_folder = './'
    # save_array_of_images_as_jpg(store_folder, name, mouth_frames)


    # there are 54 json files from s2 to s55
    for i in range(index_first_speaker, index_last_speaker + 1, 1):
        print('Processing speaker ' + str(i))
        # compute the path of the json file
        json_file_path = JSON_FOLDER + 's' + str(i) + '.json'

        # open the file
        with open(json_file_path) as json_file:
            data = json.load(json_file)
            # for each file, loop over all the recordings
            for recording in data:
                spkr = recording['SPKR']
                cond = recording['COND']
                utterance = recording['UTTERANCE']

                if recording['STATUS'] == 'CORRECT':
                    # compute the path of the video recording
                    video_path = VIDEO_FOLDER + spkr + '_' + cond + '_' + utterance + '.mov'
                    # compute path of the associated json alignment file
                    align_path = ALIGN_FOLDER + spkr + '_' + cond + '_' + utterance + '.json'
                    #
                    name = spkr + '_' + cond + '_' + utterance

                    # compute indices of beginning and end of the first word
                    idx_beg, idx_end, label_of_word = extract_frame_indices_first_word(video_path, align_path, name)

                    # compute the path to store the results
                    store_folder = RESULT_FOLDER + SPLIT + label_of_word + '/' + name + '/'    # e.g './lombardgrid/one_word_dataset/'

                    # compute the cropped images of the mouth
                    mouth_frames, mouth_shapes = extract_mouth_video(video_path, FACE_PREDICTOR_PATH)
                    mouth_frames = mouth_frames[idx_beg:idx_end]
                    # save the images to the results folder
                    save_array_of_images_as_jpg(store_folder, name, mouth_frames)

                    # convert mouth shape array to tensor
                    mouth_shapes = torch.tensor(mouth_shapes)
                    save_mouth_shape_to_file(store_folder, name, mouth_shapes)

                    

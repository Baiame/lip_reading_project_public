import os
import json
import cv2
from video_processing import extract_mouth_video
from video_processing import save_array_of_images_as_jpg
from video_processing import save_mouth_shape_to_file
import torch

FACE_PREDICTOR_PATH = './shape_predictor_68_face_landmarks.dat'
ALIGN_FOLDER = './lombardgrid_2_high_res/alignment/'
VIDEO_FOLDER= './lombardgrid_2_high_res/front/'
RESULT_FOLDER = './lombardgrid_2_high_res/12_words_dataset/'
SPLIT = 'train/'

FPS = 25
INDEX_FIRST_SPEAKER = 1 #2              ==> do s5 for valid
INDEX_LAST_SPEAKER = 4 #55


######################################################################################################


def read_align(path_to_align=None):
    """
    Extract data from alignment file
    """
    with open(path_to_align, 'r') as f:
        lines = f.readlines()	

    align = [(int(y[0])/1000, int(y[1])/1000, y[2]) for y in [x.strip().split(" ") for x in lines]]
    return align


def extract_frame_indices_first_word(path_to_align):
    """
    Extract the start and end frame index of the first word for a particular alignment file
    (for GRID, those words are bin/lay/place/set)
    Input : string path to alignment file
    """
    align = read_align(path_to_align)
    start_frame, end_frame, word = align[1]
    print(word)
    return int(start_frame), int(end_frame) + 1, word


def extract_frame_indices_first_second_last_words(path_to_align):
    """
    Extract the start and end frame index of the first word for a particular alignment file
    (for GRID, those words are bin/lay/place/set, blue/green/red/white, again/now/please/soon)
    Input : string path to alignment file
    """
    align = read_align(path_to_align)
    start_frames = []
    start_frames.append(int(align[1][0]))
    start_frames.append(int(align[2][0]))
    start_frames.append(int(align[6][0]))

    end_frames = []
    end_frames.append(int(align[1][1]) + 1)
    end_frames.append(int(align[2][1]) + 1)
    end_frames.append(int(align[6][1]) + 1)

    words = []
    words.append(str(align[1][2]))
    words.append(str(align[2][2]))
    words.append(str(align[6][2]))

    return start_frames, end_frames, words


def load_data(index_first_speaker, index_last_speaker):
    """
    Extract and store the images of the first words of the dataset sentences
    """
    for i in range(index_first_speaker, index_last_speaker + 1, 1):
        print('Processing speaker {}'.format(i))
        speaker_folder = VIDEO_FOLDER + 's' + str(i) + '/'
        files = os.listdir(speaker_folder)
        count = 0
        for video_file in files:
            count += 1
            print('Video file {}/1000'.format(count))
            video_name = os.path.splitext(video_file)[0]
            extension = os.path.splitext(video_file)[1]
            # verify it is a video
            if extension == '.mpg':
                video_path = speaker_folder + video_file
                align_path = ALIGN_FOLDER + video_name + '.align'
                start_frames, end_frames, words = extract_frame_indices_first_second_last_words(align_path)

                for k in range(3):

                    store_folder = RESULT_FOLDER +  SPLIT + words[k] + '/' + video_name + '/'
                    print(words[k])

                    # extract the indices and then the frames
                    mouth_frames, mouth_shapes = extract_mouth_video(video_path, FACE_PREDICTOR_PATH)
                    mouth_frames = mouth_frames[start_frames[k]:end_frames[k]]
                    save_array_of_images_as_jpg(store_folder, video_name, mouth_frames)



if __name__ == '__main__':
    #start_frame, end_frame, word = extract_frame_indices_first_word('./lombardgrid_2/alignment/bbae8n.align')
    load_data(INDEX_FIRST_SPEAKER, INDEX_LAST_SPEAKER)
    print(0)
import os
import json
from video_processing import extract_mouth_video
from video_processing import save_array_of_images_as_jpg

FACE_PREDICTOR_PATH = './shape_predictor_68_face_landmarks.dat'
JSON_FOLDER = './lombardgrid/json/'
VIDEO_FOLDER= './lombardgrid/front/'
RESULT_FOLDER = './lombardgrid/mouths/'

index_first_speaker = 2
index_last_speaker = 55

if __name__ == "__main__":
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
                # compute name of the recording
                name = recording['SPKR'] + '_' + recording['COND'] + '_' + recording['UTTERANCE'] 
                # compute the path of the video recording
                video_path = VIDEO_FOLDER + name + '.mov'
                # compute the path to store the results
                store_folder = RESULT_FOLDER + 's' + str(i) + '/' + name + '/'       # e.g './lombardgrid/mouths/s36/'
                # compute the cropped images of the mouth
                mouth_frames = extract_mouth_video(video_path, FACE_PREDICTOR_PATH)
                # save the images to the results folder
                save_array_of_images_as_jpg(store_folder, name, mouth_frames)

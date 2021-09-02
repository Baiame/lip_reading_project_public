# import the necessary packages
from imutils import face_utils
import math
import sys
import select
import imutils
import dlib
import cv2
import numpy as np
import time
from threading import Thread

import torch
from dataset_transformers import LipsDataset
from model_dense_transformer_bigru import LipReadingNetwork
import torch.nn as nn
import editdistance

from pynput import keyboard
from array_text_conversion import *



#########################################################################


PATH_TO_SHAPE_PREDICTOR = "./sentence_lip_reading/shape_predictor_68_face_landmarks.dat"

triggered = False

dict_labels = {
    'verb' : ['BIN', 'LAY', 'PLACE', 'SET'],
    'colour' : ['BLUE', 'GREEN', 'RED', 'WHITE'],
    'prep' : ['AT', 'BY', 'IN', 'WITH'],
    'letter' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R','S', 'T', 'U', 'V', 'X', 'Y', 'Z'],
    'digit' : ['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE'],
    'adverb' : ['AGAIN', 'NOW', 'PLEASE', 'SOON']
}




#########################################################################



class FrameAccumulator():
    """
    When triggered, can be used to store the frames of the webcam.
    """
    def __init__(self):
        self.num_frames = 0
        self.video = []
        self.triggered = False
        self.start_time = None

    def start(self):
        self.triggered = True
        self.start_time = time.time()

    def add_frame(self, frame):
        # Add frame and update counter
        self.video.append(frame)
        self.num_frames += 1

    def count_fps(self):
        return int(self.num_frames / (time.time() - self.start_time))

    def delete(self):
        # Reinitialize video and counter
        self.triggered = False
        self.num_frames = 0
        self.video.clear()



################################################################################



def on_press(key):
    global triggered
    try:
        k = key.char  # single-char keys
        # Start prediction
        if k == 's':
            triggered = True
    except:
        k = key.name  # other keys
    print('Key pressed: ' + k)

def ctc_decode(y):
    y = y.argmax(-1)
    return [ctc_convert_array_to_text(y[_], start=1) for _ in range(y.size(0))]

def correct_prediction(prediction, dict_labels):
    """
    Correct the prediction using the words of the dictionnary
    """
    new_pred = []
    list_pred_words = prediction.split(" ")
    idx_dict = 0
    # Do the correction for all the words in the prediction sentence
    for pred in list_pred_words:
        lowest_cer = math.inf
        idx_lowest_label = 0
        # For each word, find the label word with the smallest CER
        for idx, label in enumerate(dict_labels[list(dict_labels.keys())[idx_dict]]):
            cer = 1.0 * editdistance.eval(pred, label)
            if cer < lowest_cer:
                idx_lowest_label = idx
                lowest_cer = cer
        new_pred.append(dict_labels[list(dict_labels.keys())[idx_dict]][idx_lowest_label])
        idx_dict += 1
    return ' '.join(new_pred)




#################################################################################################################
#################################################################################################################
#################################################################################################################




if __name__ == '__main__':
    # Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
    print("[INFO] Loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PATH_TO_SHAPE_PREDICTOR)

    # Load the model
    print("[INFO] Loading the model...")
    opt = __import__('options')
    model = LipReadingNetwork()
    model = model.cuda()
    net = nn.DataParallel(model).cuda()
    model_dict = model.state_dict()

    # Load the weight files
    print("[INFO] Loading the pretrained weights...")
    pretrained_dict = torch.load(opt.weights)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                            k in model_dict.keys() and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # Event listener: listen for the keyboard inputs
    listener = keyboard.Listener(on_press=on_press)
    listener.start()  # start to listen on a separate thread
    #listener.join()  # remove if main thread is polling self.keys

    frame_accumulator = FrameAccumulator()

    print("[INFO] Starting...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    show_prediction_on_screen = False
    while True:
        while frame_accumulator.num_frames < 75:

            # Read frame, resize and turn to gray
            ret, frame = cap.read()
            frame = imutils.resize(frame, width=1000)
            frame_to_display = imutils.resize(frame, width=1000)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the grayscale frame
            rects = detector(gray, 0)

            # Loop over the face detections
            for rect in rects:
                # Determine the facial landmarks for the face region
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                # only retrieve the coordinates of the mouth points
                mouth = shape[48:68]

                # Loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
                for (x, y) in shape:
                    cv2.circle(frame_to_display, (x, y), 3, (0, 0, 255), -1)

            
            # If the user has pressed the key, start accumulatiing the frames
            if triggered:
                triggered = False
                frame_accumulator.start()
                print('[INFO] Starting frame accumulator...')

            if frame_accumulator.triggered == True:
                # extract the ROI (mouth) from the frame with the extreme coordinates
                (x, y, w, h) = cv2.boundingRect(np.array([mouth]))
                mouth_roi = frame[y - 50 : y + h + 50, x - 50 : x + w + 50]
                mouth_roi = cv2.resize(mouth_roi, (128,64), interpolation=cv2.INTER_CUBIC)
                frame_accumulator.add_frame(mouth_roi)
            

            # show the frame
            if show_prediction_on_screen:
                cv2.putText(frame_to_display, corrected_pred_text, (300,700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow("Frame", frame_to_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Once the accumulator is full
        print(frame_accumulator.start_time)
        fps = frame_accumulator.count_fps()
        print('[INFO] Frames loaded, FPS : {}'.format(fps))
        print("[INFO] Calling the model...")
        video =   torch.FloatTensor(frame_accumulator.video).cuda()
        video = video.permute(3, 0, 1, 2)
        video = video.reshape(1, 3, 75, 64, 128)

        with torch.no_grad():
            # Use the model to guess the sentence
            output = net(video)
            pred_txt = ctc_decode(output)[0]
            print('Prediction :' )
            print(pred_txt)
            show_prediction_on_screen = True

            print('Corrected Prediction :' )
            corrected_pred_text = correct_prediction(pred_txt, dict_labels)
            print(corrected_pred_text)


        #for i, k in enumerate(frame_accumulator.video):
            #cv2.imwrite('./lipreading_with_transformers/{}.jpg'.format(i),k)

        frame_accumulator.delete()
        #time.sleep(0)



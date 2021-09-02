import os
import numpy as np
import torch
import torch.nn as nn
import cv2

from torchsummary import summary

# Import Dataset and Model
from model_dense_transformer_bigru import LipReadingNetwork
from predict_live import correct_prediction
from video_processing import *
from array_text_conversion import *
from metrics import *
from statistics import mean


##########################################################################
###################### PATHS TO CHANGE IF NEEDED #########################
##########################################################################

# Path for the videos
list_paths = [  './sentence_lip_reading/videos_for_demo/id2_vcd_swwp2s.mpg',
                './sentence_lip_reading/videos_for_demo//id23_vcd_priazn.mpg'
            ]

# List the actual spoken sentences in the videos above
truth_list = ['SET WHITE WITH P TWO SOON', 'PLACE RED IN A ZERO NOW']

# Path to the shape predictor
shape_predictor_path = './sentence_lip_reading/shape_predictor_68_face_landmarks.dat'

##########################################################################
##########################################################################
##########################################################################


dict_labels = {
    'verb' : ['BIN', 'LAY', 'PLACE', 'SET'],
    'colour' : ['BLUE', 'GREEN', 'RED', 'WHITE'],
    'prep' : ['AT', 'BY', 'IN', 'WITH'],
    'letter' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R','S', 'T', 'U', 'V', 'X', 'Y', 'Z'],
    'digit' : ['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE'],
    'adverb' : ['AGAIN', 'NOW', 'PLEASE', 'SOON']
}

def decode_ctc(array):
    # Convert one hot encoder to tensor with corresponding index
    # (B, T, Emb=28) --> (B, T)
    array = array.argmax(-1)
    return [ctc_convert_array_to_text(array[_], start=1) for _ in range(array.size(0))]

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


if (__name__ == '__main__'):
    opt = __import__('options')

    if opt.model == 'transformer':
        model = LipReadingNetwork(transformer=True)
    elif opt.model == 'bigru':
        model = LipReadingNetwork(transformer=False, lstm=False, bidir=True)
    else:
        print('ERROR: cannot recognise model in option.py')
    model = model.cuda()

    # Uncomment to see the model structure
    #summary(model, (3, 75, 64, 128))

    
    net = nn.DataParallel(model).cuda()
    model_dict = model.state_dict()

    # Load the weight files
    pretrained_dict = torch.load(opt.weights)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                        k in model_dict.keys() and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


    wer_list = []
    cer_list = []

    for i, video_path in enumerate(list_paths):
        

        # Retrieve video and process mouth cropping
        np_vid, _ = extract_mouth_video(video_path=video_path, face_predictor_path=shape_predictor_path)
        vid = [cv2.resize(im, (128, 64), interpolation=cv2.INTER_LANCZOS4) for im in np_vid]

        # Convert to tensor and reshape to feed the model
        vid = torch.FloatTensor(vid).cuda()
        vid = vid.permute(3, 0, 1, 2)
        vid = vid.reshape(1, 3, 75, 64, 128)

        
        with torch.no_grad():
            y = net(vid)
            print('Output size')
            print(y.size())
            pred_txt = decode_ctc(y)[0]
            corrected_pred_text = correct_prediction(pred_txt, dict_labels)

            # Print to screen
            print('Raw prediction : ' + pred_txt)
            print('Corrected prediction : ' + corrected_pred_text)

            # Compute the WER and the CER
            wer = compute_wer([corrected_pred_text], [truth_list[i]])[0]
            wer_list.append(wer)
            cer = compute_cer([corrected_pred_text], [truth_list[i]])[0]
            cer_list.append(cer)
        
        blank_image = np.zeros((450, 1000, 3), np.uint8)
        cv2.putText(blank_image, 'Raw Prediction: ' + pred_txt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(blank_image, 'Corrected Prediction: ' + corrected_pred_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(blank_image, 'Truth: ' + truth_list[i], (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(blank_image, 'Mean WER = {} and Mean CER = {} since beginning'.format(round(mean(wer_list), ndigits=3), round(mean(cer_list), ndigits=3)), (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow('Prediction', blank_image)
        cv2.waitKey(0)

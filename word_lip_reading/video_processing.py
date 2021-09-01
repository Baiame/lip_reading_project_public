# import the necessary packages
import os
import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from imutils.video import VideoStream
import torch

# I have two possibilities for now :
#       - extract the mouth shape and process it
#       - extract mouth video and process it

def extract_mouth_shape(video_path, face_predictor_path): #TODO: normalize the output ?
    """
    Takes as input the path of the video and the path of the predictor and outputs an array of the mouth shape for all the frames.
    """
    # list that will contain the mouth shape for all the frames
    mouth_shapes = []

    # initialize face detector
    # create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_predictor_path)

    # import the video
    video = cv2.VideoCapture(video_path)

    # loop over the frames from the video
    while (video.isOpened()):

        ret, frame = video.read()

        # if the frame is correctly read
        if ret:
            # resize the video with a width of 600 pixels
            # convert the video to grayscale
            frame = imutils.resize(frame, width=1000)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detect faces in the grayscale frame
            rects = detector(gray_frame, 0)

            # loop over the face detections
            for rect in rects:
                # determine the facial landmarks for the face region
                shape = predictor(gray_frame, rect)
                # convert the facial landmark (x, y)-coordinates to array
                shape = face_utils.shape_to_np(shape)
                # only retrieve the coordinates of the mouth points
                mouth = shape[48:68]
                # store the mouth point for each frame
                mouth_shapes.append(mouth)
        
        # exit when ret is False (no more frames)
        else:
            break

    video.release()
    return np.array(mouth_shapes)



def extract_mouth_video(video_path, face_predictor_path):
    """
    Takes as input the path of the video and the path of the predictor and outputs an array containing the cropped
    images of the mouth in the video: extract the ROI
    Output image size : 224*224
    """
    # list that will contain the mouth shape for all the frames
    mouth_frames = []
    mouth_shapes = []

    # initialize face detector
    # create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_predictor_path)

    # import the video
    video = cv2.VideoCapture(video_path)

    # loop over the frames from the video
    while (video.isOpened()):

        ret, frame = video.read()

        # if the frame is correctly read
        if ret:
            # resize the video with a width of 1000 pixels, if less: poor quality = bad detection
            # convert the video to grayscale
            frame = imutils.resize(frame, width=1000)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detect faces in the grayscale frame
            rects = detector(gray_frame, 0)

            # loop over the face detections
            for rect in rects:
                # determine the facial landmarks for the face region
                shape = predictor(gray_frame, rect)
                # convert the facial landmark (x, y)-coordinates to array
                shape = face_utils.shape_to_np(shape)
                # only retrieve the coordinates of the mouth points
                mouth = shape[48:68]

                # extract the ROI (mouth) from the frame with the extreme coordinates
                (x, y, w, h) = cv2.boundingRect(np.array([mouth]))
                
                # refactor the coordinates of the mouth because we are going to resize the frame
                mouth[:,0] = mouth[:,0] - x
                mouth[:,1] = mouth[:,1] - y

                # store the shape of the mouth
                mouth_shapes.append(mouth)
                # Uncomment this to show the dots on the image
                #for (x1, y1) in mouth:
                    #cv2.circle(frame, (x1, y1), 1, (0, 0, 255), -1)

                mouth_roi = frame[y - 50 : y + h + 50, x - 50 : x + w + 50]
                #mouth_roi = imutils.resize(mouth_roi, width=300, height=150, inter=cv2.INTER_CUBIC)
                mouth_roi = cv2.resize(mouth_roi, (160,80), interpolation=cv2.INTER_CUBIC)
                # store the cropped mouth
                mouth_frames.append(mouth_roi)

                # Uncomment this to show the cropped mouth
                #cv2.imshow("ROI", mouth_roi)
                #cv2.waitKey(0)

        # exit when ret is False (no more frames)
        else:
            break

    video.release()
    return np.array(mouth_frames), np.array(mouth_shapes)


def save_array_of_images_as_jpg(store_path, image_generic_name, mouth_frames):
    """
    Take as input the result path and the generic name for the images, and an array of images
    """
    if len(mouth_frames) == 0:
        print('Cannot compute extraction: mouth not detected')
    # Loop over all the frames of the array
    for (i, image) in enumerate(mouth_frames):
        # Create directory if it does not exist and only if the extraction went well
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        # Save the image as .png
        cv2.imwrite(store_path + image_generic_name + '_' + "%d"%(i+1) + ".jpg", image)

def save_mouth_shape_to_file(store_path, file_name, mouth_shapes):
    """
    Take as input the result path and the name of the shape file, and tensor of shapes
    """
    # if the folder exists, it means that the mouth frames have been correctly extracted
    if os.path.exists(store_path):
        path = store_path + file_name + '.pt'
        torch.save(mouth_shapes, path)

if __name__ == '__main__':
# Test
#l = extract_mouth_shape(video_path='./lombardgrid/front/s2_l_bbim3a.mov', face_predictor_path='./shape_predictor_68_face_landmarks.dat')
#print(np.shape(l))


    e, f = extract_mouth_video(video_path='C:/Users/aurel/Downloads/grid_dataset_downloaded/test_videos/bin_blue_with_Y7_again/WIN_20210823_15_09_00_Pro.mp4', face_predictor_path='./shape_predictor_68_face_landmarks.dat')
    save_array_of_images_as_jpg('C:/Users/aurel/Downloads/grid_dataset_downloaded/test_videos/bin_blue_with_Y7_again/', 'aurele', e)
# Automated Lip Reading
## Description of the project
This project aims to develop and test different lip reading algorithms on words and on sentences, using the GRID Corpus Dataset.
- Link to the project: https://github.com/Baiame/lip_reading_project_public.
- Weights and predictor can be found at this address : https://drive.google.com/drive/folders/13qP89f225QIiHfvf7LUDiWH6aREzjew5?usp=sharing
- The GRID Corpus dataset can be downloaded at this address: http://spandh.dcs.shef.ac.uk/gridcorpus/
------------------------------------
## Requirements
- pytorch (CUDA version)
- tensorboard
- opencv
- numpy
- editdistance
- sklearn

------------------------------------
## Sentence Lip Reading
### Description of the files
|Python File|Usage|
|--- | ---|
|array_text_conversion.py|Contains the functions used to convert CTC array to text|
|create_the_index_txt_file.py|Compute the text files containing the path|
|dataset_transformers.py|Dataset implementation|
|evaluate_model.py|Run evaluation on the given model and create confusion matrices|
|metrics.py|Contains the metric functions|
|model_dense_transformer_bigru.py|Implementation of the sentence lip reading model|
|dense_modules.py|Dense3D modules to construct the front-end network|
|options.py|Parameters file|
|predict.py|Run the demo|
|predict_live.py|Run the live prediction using the webcam|
|train_model.py|Training|
|video_processing.py|Contain data processing functions to create the pre-processed GRID dataset|

### How to run
#### Demo
- Download the weights
- Add the _shape_predictor_68_face_landmarks.dat_ file in the _sentence_lipreading_ folder if missing.
- Modify the weights path and the model type in the _options.py_ file if needed.
- Run _predict.py_ for the demo.
(Run _predict_live.py_ for live  using the webcam)

#### Training
- To train a new model, comment the _weights_ line in _options.py_ otherwise it will continue to train the existing model under the _weights_ path.
- Download and extract _fraction_processed_dataset_slr_ from the given link above. It contains the lips frames, the alignment files and the txt files that contain the paths of the videos for training and validation.
- Modify, in the _options.py_ file, the video_path (links to _lips_ folder), the alignment_path (links to _alignment_ folder),the training_list (links to _video_paths_list_training.txt_) and the validation_list (links to _video_paths_list_validation.txt_). These paths must link to the dataset folder, and the train and validation text files. Modify save_model_path as well, to where you want.
- Run train_model.py

-----------------------------------
## Word Lip Reading
|Python File|Usage|
|--- | ---|
|dataset_mouth_shape.py|Dataset object of the mouth shapes|
|dataset_one_word.py|Dataset object of the one-word long videos|
|model_mouth_shape.py|Implementation of the mouth shapes model|
|model_one_word.py|Implementation of the one-word long videos model|
|test.py|Run the test of the model|
|train_one_word.py|Train a model on the chosen dataset|
|video_processing.py|Contain video processing tools|
|compute_images_from_video.py|Compute the one-word long videos dataset given the complete GRID Corpus dataset|
|compute_images_from_videos_light.py|Compute a light version of the one-word long videos dataset given the complete GRID Corpus dataset|

### How to run
#### Testing
Change path of weights file and of the test files
Run _test.py_
#### Training
Compute the dataset
Run _train_one_word.py_


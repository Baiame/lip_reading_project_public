# Idx of GPU to use with CUDA
gpu = '0'
# Set a non random seed to be able to reproduce the results
seed = 0

# Training method: unseen or overlapped
training_method = 'unseen'

# Name of the model
model_name = 'dense3d_transformer'

# Path to the folder that contains the lip videos (frames)
video_path = 'C:/Users/aurel/Downloads/grid_dataset_downloaded/lip/'

# Paths to the txt file that contains the path of the folders for training and validation
training_list = 'C:/Users/aurel/Downloads/grid_dataset_downloaded/scenario_unseen_train_list.txt'
validation_list = 'C:/Users/aurel/Downloads/grid_dataset_downloaded/scenario_unseen_val_list.txt'

# Path to the folder that contains the alignment files
alignment_path = 'C:/Users/aurel/Downloads/grid_dataset_downloaded/GRID_align_txt/'

# Padding value for the video and the text
padding_video = 75
padding_text = 50

# Batch size (max 16 for RTX2070)
batch_size = 16

# Learning rate
learning_rate = 2e-5
decay = 0
momentum=0.9

# Stop the training
max_epoch = 200

# Display training process to screen
display = 1

# Test and save new model very ... steps
test_step = 2000

# Path to save the model
save_model_path = f'C:/Users/aurel/Downloads/grid_dataset_downloaded/trained_models/model_{training_method}_{model_name}'

# If true: Optimize the model that is located under the 'weights' path or a new model if no path is given
optimize = True

# Comment these 2 lines if you want to train a new model
model = 'transformer'
weights = './sentence_lip_reading/weights/model_unseen_dense3d_transformer_loss_0.7470988631248474_wer_0.2406666666666667_cer_0.1106539869955016.pt'
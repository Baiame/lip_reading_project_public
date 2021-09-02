import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from dataset_transformers import LipsDataset
import numpy as np
import torch.nn as nn
from model_dense_transformer_bigru import LipReadingNetwork
from tensorboardX import SummaryWriter
from torchsummary import summary
from metrics import *
from array_text_conversion import *


if __name__ == '__main__':
    options = __import__('options')    
    summary_writer = SummaryWriter()

# Show intermediate results
show = False


###################################################################################
def decode_ctc(array):
    # Convert one hot encoder to tensor with corresponding index
    # (B, T, Emb=28) --> (B, T)
    array = array.argmax(-1)
    length = array.size(0)
    text_array = []
    for _ in range(length):
        text_array.append(ctc_convert_array_to_text(array[_], start=1))
    return text_array

def create_folder_if_not_existing(save_path):
    (path, _) = os.path.split(save_path)
    if(not os.path.exists(path)):
        os.makedirs(path)

def load_model(path_weights, model):
    print("Loading weights...")
    # Retrieve the weights
    loaded_weights = torch.load(path_weights)
    model_weight = model.state_dict()
    loaded_weights = {n: m for n, m in loaded_weights.items() if n in model_weight.keys() and m.size() == model_weight[n].size()}

    # Update and load into the model
    model_weight.update(loaded_weights)
    model.load_state_dict(model_weight)
    return model

###################################################################################

def validation(model, network):
    """ Evaluate the given model on the validation dataset"""

    # Deactivate gradient optimization
    with torch.no_grad():

        # Load validation dataset and create Dataloader
        dataset = LipsDataset(options.alignment_path,
                                options.validation_list,
                                options.padding_video,
                                options.padding_text,
                                'test')

        validation_loader = DataLoader( dataset,
                                        batch_size = options.batch_size,
                                        shuffle=False,
                                        num_workers=0,
                                        drop_last = False)
        # Set evaluation mode  
        model.eval()

        # Initialize storage lists
        loss_list = []
        wer_list = []
        cer_list = []

        # Initialize the CTC loss
        criterion = nn.CTCLoss()

        print('Number of test data:{}'.format(len(dataset.video_speaker_name)))
        
        # Loop over the validation loader
        for (iteration, input) in enumerate(validation_loader):

            # Load video to GPU        
            video = input.get('video').cuda()             # (B, C, T, H, W)
            

            # Load label and lengths
            txt_label = input.get('text').cuda()

            if show == True:
                print('Text label : {}'.format(txt_label))

            # Pass video through network
            output = network(video)

            # Get the lengths
            video_length = input.get('video_length').cuda()
            txt_length = input.get('text_length').cuda()
            if show == True:
                print('Lengths : Video = {} / Text = '.format(video_length, txt_length))
            
            # Compute loss
            loss = criterion(output.transpose(0, 1).log_softmax(-1), txt_label, video_length.view(-1), txt_length.view(-1))

            # Detach loss to CPU
            loss = loss.detach().cpu().numpy()
            loss_list.append(loss)

            # Translate prediction and true label into text
            predicted_txt = decode_ctc(output)
            truth_txt = []
            for _ in range(txt_label.size(0)):
                truth_txt.append(convert_array_to_text(txt_label[_], start=1))


            # Compute WER and CER to lists
            wer = compute_wer(predicted_txt, truth_txt)
            cer = compute_cer(predicted_txt, truth_txt)

            # Append WER and CER to lists
            wer_list.extend(wer) 
            cer_list.extend(cer)

            # Compute mean WER and mean CER
            mean_wer = np.array(wer_list).mean()
            mean_cer = np.array(cer_list).mean()

            # Display to screen              
            if(iteration % options.display == 0):
                print('---------------------------------------------------------------------------------')                
                print('{:<40}|{:>40}'.format('Prediction', 'Truth'))
                print('---------------------------------------------------------------------------------')
                zip_list =  list(zip(predicted_txt, truth_txt))[:10]               
                for (predicted, true_sentence) in zip_list:
                    print('{:<40}|{:>40}'.format(predicted, true_sentence))                
                print('---------------------------------------------------------------------------------')
                print('Validation: Iterations = {}, WER = {}, CER = {}'.format(iteration, mean_wer, mean_cer))                
                print('---------------------------------------------------------------------------------')

            
                
        return (np.array(loss_list).mean(), mean_wer, mean_cer)



def train(model, network):
    """ Launch the training process, using a model and a DataParallel object (network). Inspired from LipNet
    """

    torch.manual_seed(options.seed)
    torch.cuda.manual_seed_all(options.seed)

    # Initialize Loss
    criterion = nn.CTCLoss()
    # Load the dataset

    train_dataset = LipsDataset(options.alignment_path,
                                options.training_list,
                                options.padding_video,
                                options.padding_text,
                                'train')

    # Create the train DataLoader with the dataset
    train_loader = DataLoader(  train_dataset,
                                batch_size = options.batch_size,
                                num_workers=0,
                                shuffle=True,
                                drop_last = False)

    len_dataset = len(train_dataset.video_speaker_name)
   

                
    print('Amount of training videos: ' + str(len_dataset))  
 
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr = options.learning_rate)


    train_cer = []
    train_wer = []

    # Training loop
    for epoch in range(options.max_epoch):
        for (i_iter, input) in enumerate(train_loader):

            # Set training mode
            model.train()

            # Load the video on GPU
            video = input.get('video').cuda()             # (B, C, T, H, W)
            
            # Reset gradient
            optimizer.zero_grad()

            # Load label and lengths on GPU
            txt_label = input.get('text').cuda()
            if show == True:
                print('Text label : {}'.format(txt_label))

            # Output prediction for given input
            output_network = network(video)

            # Get the lengths
            video_length = input.get('video_length').cuda()
            txt_length = input.get('text_length').cuda()
            if show == True:
                print('Lengths : Video = {} / Text = '.format(video_length, txt_length))

            # Calculate loss and update network
            loss = criterion(output_network.transpose(0, 1).log_softmax(-1), txt_label, video_length.view(-1), txt_length.view(-1))
            loss.backward()
            
            

            # Compare the outputed text and the truth
            predicted_txt = decode_ctc(output_network)

            truth_txt = []
            for _ in range(txt_label.size(0)):
                truth_txt.append(convert_array_to_text(txt_label[_], start=1))

            # Compute WER
            wer = compute_wer(predicted_txt, truth_txt)
            cer = compute_cer(predicted_txt, truth_txt)

            # Append to lists
            train_wer.extend(wer)
            train_cer.extend(cer)


            # Compute mean WER and mean CER
            mean_wer = np.array(train_wer).mean()
            mean_cer = np.array(train_cer).mean()
                        
            if(options.optimize):
                optimizer.step()

            total_iterations = i_iter + epoch * len(train_loader)
            # Display on the screen
            if(total_iterations % options.display == 0):
                
                summary_writer.add_scalar('Training Loss', loss, total_iterations)
                summary_writer.add_scalar('Training WER', np.array(train_wer).mean(), total_iterations)              
                print('---------------------------------------------------------------------------------')                
                print('{:<40}|{:>40}'.format('Prediction', 'Truth'))                
                print('---------------------------------------------------------------------------------')
                
                zip_list = list(zip(predicted_txt, truth_txt))[:3]
                for (prediction, truth) in zip_list:
                    print('{:<40}|{:>40}'.format(prediction, truth))
                print('---------------------------------------------------------------------------------')               
                print('Epoch = {}, Total Iterations = {}, Loss = {}, Train WER = {}, Train CER = {}'.format(epoch, total_iterations, loss, mean_wer, mean_cer))
                print('---------------------------------------------------------------------------------')
            

            # Evaluate the model
            if(total_iterations % options.test_step == 0):
                
                # Launch evaluation              
                (evaluation_loss, evaluation_wer, evaluation_cer) = validation(model, network)

                # Print results to the screen
                print('Iterations = ' + str(total_iterations) + ', Loss = ' + str(evaluation_loss) + ', WER = ' + str(evaluation_wer) + ', CER = ' + str(evaluation_cer))                  

                # Save the model
                save_path = '{}_loss_{}_wer_{}_cer_{}.pt'.format(options.save_model_path, evaluation_loss, evaluation_wer, evaluation_cer)
                create_folder_if_not_existing(save_path)
                torch.save(model.state_dict(), save_path)

                # Save for tensorboard
                summary_writer.add_scalar('Validation WER', evaluation_wer, total_iterations)
                summary_writer.add_scalar('Validation CER', evaluation_cer, total_iterations)
                summary_writer.add_scalar('Validation Loss', evaluation_loss, total_iterations)  

                # If the goal is not to optimize the model: exit the training
                if(not options.optimize):
                    exit()



if __name__ == '__main__':
    print("Creating the model...")
    model = LipReadingNetwork(transformer=True)
    model = model.cuda()
    print("Model architecture : ")
    summary(model, (3, 75, 64, 128))
    network = nn.DataParallel(model).cuda()

    # Load the weight if the file exists
    weight_exists = hasattr(options, 'weights')
    if(weight_exists == True):
        model = load_model(options.weights, model)

    print("The training begins")
    train(model, network)
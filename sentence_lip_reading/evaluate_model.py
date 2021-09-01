import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from predict_live import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from metrics import *
from array_text_conversion import *




# Import Dataset and Model
from dataset_transformers import LipsDataset
from model_dense_transformer_bigru import LipReadingNetwork

# Imports for monitoring and plots
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix

if (__name__ == '__main__'):
    opt = __import__('options')
    summary_writer = SummaryWriter()



###################################################################################
def decode_ctc(array):
    # Convert one hot encoder to tensor with corresponding index
    # (B, T, Emb=28) --> (B, T)
    array = array.argmax(-1)
    length = array.size(0)
    return [ctc_convert_array_to_text(array[_], start=1) for _ in range(length)]

###################################################################################


def evaluate_model(model, net):
    with torch.no_grad():
        # Load dataset
        test_dataset = LipsDataset(opt.alignment_path,
                                opt.validation_list,
                                opt.padding_video,
                                opt.padding_text,
                                'test')

        print('Number of test data: {}'.format(len(test_dataset.video_speaker_name)))

        # Set model to evaluation mode
        model.eval()

        # Create the test loader
        test_loader = DataLoader(test_dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=0,
                            drop_last=False)

        # Initialisation of the storage lists
        wer_list = []
        cer_list = []

        predicted_labels = []
        truth_labels = []

        for (iteration, input) in enumerate(test_loader):
            
            # Get video and label and upload on GPU
            video = input.get('video').cuda()
            txt_label = input.get('text').cuda()

            # Forward pass
            output = net(video)

            # Compare the prediction and the truth
            predicted_text = decode_ctc(output)
            truth_text = [convert_array_to_text(txt_label[_], start=1) for _ in range(txt_label.size(0))]
            wer_list.extend(compute_wer(predicted_text, truth_text))
            cer_list.extend(compute_cer(predicted_text, truth_text))
            
            # Get all the word pairs (Corrected pred / Truth) for the confusion matrix
            corrected_pred_text = [correct_prediction(pred, dict_labels) for pred in predicted_text]
            pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(corrected_pred_text, truth_text)]
            for pair in pairs:
                pred_list = pair[0]
                truth_list = pair[1]
                for k in range(len(pred_list)):
                    predicted_labels.append(pred_list[k])
                    truth_labels.append(truth_list[k])


            # Display on screen
            if (iteration % opt.display == 0):
                # Upper part of the table
                print('-------------------------------------------------------------------------')
                print('{:<40}|{:>40}'.format('Prediction', 'Truth'))
                print('-------------------------------------------------------------------------')

                # Print 10 examples
                examples = list(zip(predicted_text, truth_text))[:10]
                for (predict, truth) in examples:
                    print('{:<40}|{:>40}'.format(predict, truth))
                
                # Lower part of the table
                print('-------------------------------------------------------------------------')
                print('Total Iterations = {}, Test WER = {}, Test CER = {}'.
                      format(iteration, np.array(wer_list).mean(), np.array(cer_list).mean()))
                print('-------------------------------------------------------------------------')

            # Add the WER and CER to tensorboard
            summary_writer.add_scalar('Test WER', np.array(wer_list).mean(), iteration)
            summary_writer.add_scalar('Test CER', np.array(cer_list).mean(), iteration)


        return np.array(wer_list).mean(), np.array(cer_list).mean(), predicted_labels, truth_labels


if __name__ == '__main__':
    print("Loading options...")
    # Create the model
    model = LipReadingNetwork()
    model = model.cuda()
    net = nn.DataParallel(model).cuda()
    model_dict = model.state_dict()

    # Load the weight files
    pretrained_dict = torch.load(opt.weights)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict.keys() and v.size() == model_dict[k].size()}

    # Update the model weights with the pretrained weights
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print("The model is loaded")
    print("The evaluation on the test set is starting")
    mean_wer, mean_cer, predicted_labels, truth_labels = evaluate_model(model, net)

    print('Test WER and CER final values, WER = {}, CER = {}'.format(mean_wer, mean_cer))


    # Create the plots
    for k in range(0,6):
        plt.clf()
        labels = dict_labels[list(dict_labels.keys())[k]]

        conf_mat = confusion_matrix(truth_labels, predicted_labels, labels=labels, normalize='true')
        conf_mat = np.around(conf_mat, decimals=2)
        df_conf_mat = pd.DataFrame(conf_mat, index=labels, columns=labels)
        plt.figure(figsize = (20, 20))
        sn.heatmap(df_conf_mat, annot=True, annot_kws={"size": 10}, cmap=plt.cm.Blues)
        plt.title('Confusion Matrix for the {}'.format(list(dict_labels.keys())[k]))
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

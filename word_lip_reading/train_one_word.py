import torch
import torch.nn.functional as F
from dataset_one_word import VideoDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import statistics
from torchvision import transforms


from model_one_word import Net
from model_one_word import Net_v2
from model_mouth_shape import Network
from dataset_mouth_shape import ShapeDataset

def my_collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

def label_to_one_hot(label):
    y_onehot = torch.zeros(4)
    y_onehot[label] = 1
    return y_onehot


if __name__ == '__main__':

    cuda = True if torch.cuda.is_available() else False

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    ##########################################
    ###### Values to change for training #####
    ##########################################

    valid_every_n_steps = 40
    mode = 'one_word_grid_dataset'

    ##########################################
    ##########################################
    ##########################################


    # Selection of the dataset
    if mode == 'one_word':

        # apply the right transformation to the images
        transforms = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # load datasets
        train_dataset = VideoDataset(root_dir='./lombardgrid/one_word_dataset/',
                                    split='train')
        train_loader = torch.utils.data.DataLoader(
                                                    dataset=train_dataset,
                                                    batch_size=32, 
                                                    shuffle=True, 
                                                    collate_fn=my_collate_fn)

        validation_dataset = VideoDataset(root_dir='./lombardgrid/one_word_dataset/', split='valid')
        validation_loader = torch.utils.data.DataLoader(
                                                    dataset=validation_dataset, 
                                                    batch_size=32, 
                                                    shuffle=True, 
                                                    collate_fn=my_collate_fn)

        # initialize the model
        model = Net_v2()

        # load model on gpu
        if torch.cuda.is_available():
            model.cuda()

        # opimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = torch.nn.CrossEntropyLoss()

        # store the loss for each epoch
        loss_list_epoch = []
        validation_accuracy_list = []
        iter_list = []

        model.train()

        iter = 0

        for epoch in range(10):
            # epoch accuracy
            epoch_correct = 0
            epoch_total = 0

            # store the losses for this epoch
            loss_list = []

            for i, (video, label) in enumerate(train_loader):
                torch.cuda.empty_cache()

                print("Epoch : " + str(epoch) + " ##### " + "Batch " + str(i))
            
                optimizer.zero_grad()

                video = pad_sequence(sequences=video, batch_first=True)

                outputs = model(video)

                # count the number of good predictions
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += label.size(0)

                if torch.cuda.is_available():
                    epoch_correct += (predicted.cpu() == label.cpu()).sum()
                else:
                    epoch_correct += (predicted == label).sum()

                # compute the loss
                label = label.cuda()
                loss = criterion(outputs, label)

                # propagate loss and update the weights
                if torch.cuda.is_available():
                    loss.cuda()
                loss.backward()
                optimizer.step()

                # store the loss for this batch
                loss_list.append(loss.item())
                iter +=1

                #if i == 1:
                    #print(outputs)
                    #print(label)
                
                if iter % valid_every_n_steps == 0:
                    with torch.no_grad():
                        correct = 0
                        total = 0
                        for i, (videos, labels) in enumerate(validation_loader):
                            videos = pad_sequence(sequences=videos, batch_first=True)
                            outputs = model(videos)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            if torch.cuda.is_available():
                                correct += (predicted.cpu() == labels.cpu()).sum()
                            else:
                                correct += (predicted == labels).sum()
                        v_accuracy = 100 * correct // total
                        print('Iteration {} ### Validation Accuracy : {}%'.format(iter, v_accuracy))
                        validation_accuracy_list.append(v_accuracy)
                        iter_list.append(iter)

            # print the training accuracy on this epoch
            t_accuracy = 100 * epoch_correct // epoch_total
            print('Accuracy epoch {} : {}'.format(epoch, t_accuracy))

            # store the mean loss for this epoch
            loss_list_epoch.append(statistics.mean(loss_list))

        
        # save the model
        torch.save(model.state_dict(), './models/model_15_speakers.pt')

        plt.clf()
        plt.plot(loss_list_epoch)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Evolution of the loss during training with mini-batches of size 16')
        plt.show()

        plt.clf()
        plt.plot(iter_list, validation_accuracy_list)
        plt.xlabel('Iteration')
        plt.ylabel('Validation Accuracy')
        plt.title('Evolution of the accuracy on the validation dataset')
        plt.show()


    elif mode == 'one_word_grid_dataset':

        # apply the right transformation to the images
        transforms = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # load datasets
        train_dataset = VideoDataset(root_dir='./lombardgrid_2_high_res/one_word_dataset/',
                                    split='train', transform=transforms)
        train_loader = torch.utils.data.DataLoader(
                                                    dataset=train_dataset,
                                                    batch_size=64,
                                                    shuffle=True, 
                                                    collate_fn=my_collate_fn)
                                                    
        validation_dataset = VideoDataset(root_dir='./lombardgrid_2_high_res/one_word_dataset/', split='valid')
        validation_loader = torch.utils.data.DataLoader(
                                                    dataset=validation_dataset, 
                                                    batch_size=64, 
                                                    shuffle=True, 
                                                    collate_fn=my_collate_fn)

        # initialize the model
        model = Net_v2()

        # load model on gpu
        if torch.cuda.is_available():
            model.cuda()

        # opimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = torch.nn.CrossEntropyLoss()

        # store the loss for each epoch
        loss_list_epoch = []
        validation_accuracy_list = []
        iter_list = []

        model.train()
        iter = 0

        for epoch in range(10):
            
            # epoch accuracy
            epoch_correct = 0
            epoch_total = 0

            # store the losses for this epoch
            loss_list = []

            for i, (video, label) in enumerate(train_loader):
                torch.cuda.empty_cache()

                print("Epoch : " + str(epoch) + " ##### " + "Batch " + str(i))
            
                optimizer.zero_grad()

                video = pad_sequence(sequences=video, batch_first=True)

                outputs = model(video)

                # count the number of good predictions
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += label.size(0)

                if torch.cuda.is_available():
                    epoch_correct += (predicted.cpu() == label.cpu()).sum()
                else:
                    epoch_correct += (predicted == label).sum()

                # compute the loss
                label = label.cuda()
                loss = criterion(outputs, label)

                # propagate loss and update the weights
                if torch.cuda.is_available():
                    loss.cuda()
                loss.backward()
                optimizer.step()

                # store the loss for this batch
                loss_list.append(loss.item())
                iter +=1

                #if i == 1:
                    #print(outputs)
                    #print(label)
                
                # Test the accuracy on the validation dataset
                model.eval()
                if iter % valid_every_n_steps == 0:
                    with torch.no_grad():
                        correct = 0
                        total = 0
                        for i, (videos, labels) in enumerate(validation_loader):
                            videos = pad_sequence(sequences=videos, batch_first=True)
                            outputs = model(videos)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            if torch.cuda.is_available():
                                correct += (predicted.cpu() == labels.cpu()).sum()
                            else:
                                correct += (predicted == labels).sum()
                        v_accuracy = 100 * correct // total
                        print('Iteration {} ### Validation Accuracy : {}%'.format(iter, v_accuracy))
                        validation_accuracy_list.append(v_accuracy)
                        iter_list.append(iter)
                model.train()

            # print the training accuracy on this epoch
            t_accuracy = 100 * epoch_correct // epoch_total
            print('Accuracy epoch {} : {}'.format(epoch, t_accuracy))

            # store the mean loss for this epoch
            loss_list_epoch.append(statistics.mean(loss_list))

        
        # save the model
        torch.save(model.state_dict(), './models/model_5_speakers_high_res.pt')

        plt.clf()
        plt.plot(loss_list_epoch)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Evolution of the loss during training with mini-batches of size 128')
        plt.show()

        plt.clf()
        plt.plot(iter_list, validation_accuracy_list)
        plt.xlabel('Iteration')
        plt.ylabel('Validation Accuracy')
        plt.title('Evolution of the accuracy on the validation dataset')
        plt.show()



    elif mode == 'mouth_shape':
        train_dataset = ShapeDataset('./lombardgrid/one_word_dataset/')
        

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, collate_fn=my_collate_fn)
        model = Network()
        
        if torch.cuda.is_available():
            model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        #optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
        criterion = torch.nn.CrossEntropyLoss()
        #criterion = torch.nn.NLLLoss()

        # store the loss for each epoch
        loss_list_epoch = []
        losses = []

        for epoch in range(1000):

            print("Epoch : " + str(epoch))

            #store the losses for this epoch
            loss_list = []

            for i, (shapes, label) in enumerate(train_loader):

                if torch.cuda.is_available():
                    #shapes = Variable(shapes.cuda())
                    label = Variable(label.cuda())
                # else:
                #     video = Variable(video)
                #     label = Variable(label)


            
                optimizer.zero_grad()

                shapes = pad_sequence(sequences=shapes, batch_first=True)
                shapes = Variable(shapes.cuda())

                outputs = model(shapes)

                if epoch % 10 == 0 and i == 29:
                    print(outputs)
                    print(label)

                #label = label.cuda()
                loss = criterion(outputs, label)
                #print("Loss : ", loss.item())

                if torch.cuda.is_available():
                    loss.cuda()

                loss.backward()

                optimizer.step()

                # store the loss for this batch
                loss_list.append(loss.item())
                losses.append(loss.item())

            # store the mean loss for this epoch
            loss_list_epoch.append(statistics.mean(loss_list))

        #plt.plot(losses)
        plt.plot(loss_list_epoch)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Evolution of the losss during training with mini-batches of size 32')
        plt.show()
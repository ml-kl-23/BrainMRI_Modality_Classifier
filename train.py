"""
For  torch==1.13.0cu117, torchvision==0.14.0cu117, tensorflow==2.11.0

Make directories for:
    
Models/
Results


Usage :
    
CUDA_VISIBLE_DEVICES=0 python v6_2.py --epochs=100 --learning_rate=0.0009 --p=0.028 --hidden_units=9000 


@author: Manish
"""

import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os
import tensorflow as tf

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from collections import OrderedDict
#from PIL import Image

gpu = 'cuda'
#import model_functions
#import processing_functions
torch.manual_seed(101)
torch.cuda.manual_seed(101)

import json
import argparse


parser = argparse.ArgumentParser(description='Train Image Classifier')

# Command line arguments
parser.add_argument('--arch', type = str, default = 'vgg', help = 'NN Model Architecture')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning Rate')
parser.add_argument('--hidden_units', type = int, default = 10000, help = 'Neurons in the Hidden Layer')
parser.add_argument('--epochs', type = int, default = 100, help = 'Epochs')
parser.add_argument('--gpu', type = str, default = 'cuda', help = 'GPU or CPU')
parser.add_argument('--save_dir', type = str, default = 'Models/default_model.pth', help = 'Path to checkpoint')
parser.add_argument('--p', type = float, default = 0.1, help = 'Dropout Rate')
arguments = parser.parse_args()

# Image data directories
data_dir = 'data'
train_dir = data_dir + '/TrainingSet'
valid_dir = data_dir + '/ValidationSet'
test_dir = data_dir + '/TestSet'


# Define transforms for the training, validation, and testing sets
def data_transforms():
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])])
    
    return training_transforms, validation_transforms, testing_transforms


# Load the datasets with ImageFolder
def load_datasets(train_dir, training_transforms, valid_dir, validation_transforms, test_dir, testing_transforms):
    training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transforms)
    
    return training_dataset, validation_dataset, testing_dataset


# Transforms for the training, validation, and testing sets
training_transforms, validation_transforms, testing_transforms = data_transforms()


# Load the datasets with ImageFolder
training_dataset, validation_dataset, testing_dataset =  load_datasets(train_dir, 
                                                                        training_transforms, 
                                                                        valid_dir, 
                                                                        validation_transforms, 
                                                                        test_dir, testing_transforms)


# Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
validate_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32)
test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=32)



# Build and train the neural network (Transfer Learning)
weights = models.vgg16(weights='IMAGENET1K_V1')

if arguments.arch == 'vgg':
    input_size = 25088
    model = models.vgg16(weights=weights)
elif arguments.arch == 'alexnet':
    input_size = 9216
    model = models.alexnet(pretrained=True)
    
#print(model)

# Freeze pretrained model parameters to avoid backpropogating through them
for parameter in model.parameters():
    parameter.requires_grad = False

# Build custom classifier
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, arguments.hidden_units)),
                                        ('relu', nn.ReLU()),
                                        ('drop', nn.Dropout(arguments.p)),
                                        ('fc2', nn.Linear(arguments.hidden_units, 5)),
                                        ('output', nn.LogSoftmax(dim=1))]))

model.classifier = classifier


# Function for the validation pass
def validation(model, validateloader, criterion, gpu):
    
    val_loss = 0
    accuracy = 0
    
    for images, labels in iter(validateloader):

        images, labels = images.to(gpu), labels.to(gpu)

        output = model.forward(images)
        val_loss += criterion(output, labels).item()

        probabilities = torch.exp(output)
        
        equality = (labels.data == probabilities.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return val_loss, accuracy



# Function for measuring network accuracy on test data
def test_accuracy(model, test_loader, gpu):

    # Do validation on the test set
    model.eval()
    model.to(gpu)

    with torch.no_grad():
    
        accuracy = 0
        test_accuracy = []
        y_pred= []
        y_true = []    


        for images, labels in iter(test_loader):
    
            images, labels = images.to(gpu), labels.to(gpu)
    
            output = model.forward(images)

            probabilities = torch.exp(output)
        
            equality = (labels.data == probabilities.max(dim=1)[1])
        
            accuracy += equality.type(torch.FloatTensor).mean()

            test_accuracy.append(accuracy/len(test_loader))

            pr = probabilities.max(dim=1)[1].cpu().numpy()  
            la = labels.data.cpu().numpy()
            y_pred.extend(pr)
            y_true.extend(la)
        
        
        
        print("Test Accuracy: {}".format(accuracy/len(test_loader)))  
        
        #val_accuracy.append(accuracy/len(test_loader))
        
        plt.figure(figsize=(10,5))
        plt.plot(test_accuracy,label="Test Accuracy")
        plt.xlabel("Iterations/Batch")
        plt.ylabel("Test Accuracy")
        plt.legend()
        #plt.show()   
        plt.savefig('Results/Test_Accuracy_v6.png')    

        classes = ('DSC', 'FLAIR', 'T1', 'T1ce', 'T2')    
        ## Print the Confusion Matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        print(cf_matrix)
       
        plt.figure(figsize=(10,5))
        ax = sb.heatmap(cf_matrix,fmt='d',annot=True,cmap='Blues')
        #ax.axis.set_ticklabels(['DSC','FLAIR','T1','T1ce','T2'])
        #ax.axis.set_ticklabels(['DSC','FLAIR','T1','T1ce','T2'])
        plt.savefig('Results/Conf_matrix_v6_2.png')
        print('Precision score classwise is : {} \n'.format(precision_score(y_true,y_pred, average=None)))
        print('Recall classwise is :  {} \n'.format(recall_score(y_true,y_pred,average=None)))
        print('Accuracy score is : {}\n'.format(accuracy_score(y_true,y_pred)))
        print('F1 score classwise is  : {}\n'.format(f1_score(y_true,y_pred, average=None)))

  
        
# Train the classifier
def train_classifier(model, optimizer, criterion, arg_epochs, train_loader, validate_loader, gpu):

    #with active_session():
        val_losses = []
        train_losses = []
        
        
        epochs = arg_epochs
        steps = 0
        print_every = 40

        model.to(gpu)

        for e in range(epochs):
        
            model.train()
    
            running_loss = 0
    
            for images, labels in iter(train_loader):
        
                steps += 1
        
                images, labels = images.to(gpu), labels.to(gpu)
        
                optimizer.zero_grad()
        
                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item()
        
                if steps % print_every == 0:
                
                    model.eval()
                
                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        validation_loss, accuracy = validation(model, validate_loader, criterion, gpu)
            
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Validation Loss: {:.3f}.. ".format(validation_loss/len(validate_loader)),
                          "Validation Accuracy: {:.3f}".format(accuracy/len(validate_loader)))
                    
                
                    train_losses.append(running_loss)
                    val_losses.append(validation_loss/len(validate_loader))
                    #val_accuracy.append(accuracy/len(validate_loader))
                    
                    running_loss = 0
                    model.train()   
                    
        plt.figure(1)
        plt.title("Training and Validation Loss")
        plt.plot(val_losses,label="val_loss")
        plt.plot(train_losses,label="train_loss")
        plt.plot()
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
#        plt.show()       
        plt.savefig('Results/val_loss_v6_2.png')  




# Function for saving the model checkpoint
def save_checkpoint(model, training_dataset, arch, epochs, lr, hidden_units, input_size):

    model.class_to_idx = training_dataset.class_to_idx

    checkpoint = {'input_size': (3, 224, 224),
                  'output_size': 5,
                  'hidden_layer_units': hidden_units,
                  'batch_size': 64,
                  'learning_rate': lr,
                  'model_name': arch,
                  'model_state_dict': model.state_dict(),
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  'clf_input': input_size}

    torch.save(checkpoint, 'Models/model_v6_2.pth')



# Loss function (since the output is LogSoftmax, we use NLLLoss)
criterion = nn.NLLLoss()

# Gradient descent optimizer
optimizer = optim.Adam(model.classifier.parameters(), lr=arguments.learning_rate)
    
train_classifier(model, optimizer, criterion, arguments.epochs, train_loader, validate_loader, gpu)
    
test_accuracy(model, test_loader, gpu)

save_checkpoint(model, training_dataset, arguments.arch, arguments.epochs, arguments.learning_rate, arguments.hidden_units, input_size)  








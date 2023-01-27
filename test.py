import time
start = time.time()

import warnings
warnings.filterwarnings("ignore")

import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import nibabel as nib
import dicom2jpg


import argparse

from collections import OrderedDict

parser = argparse.ArgumentParser(description='Train Image Classifier')

# Command line arguments
#parser.add_argument('--arch', type = str, default = 'vgg', help = 'NN Model Architecture only vgg16')
parser.add_argument('--mn', type = str, default ='CraiClassifier_v1.pth' , help = 'Model type ex. --mn=CraiClassifier_v1.pth')
parser.add_argument('--im', type = str, default ='_FLAIR_t2_FLAIR_spc_Ulleval_0002.jpg', help = 'Image to be classified ex. --im=_FLAIR_t2_FLAIR_spc_Ulleval_0002.jpg')
parser.add_argument('--p', type = float, default =0.10 , help = 'Dropout - v1 : 0.10, v2 :0.12 , ee. usage --p=0.12')
arguments = parser.parse_args()




def load_checkpoint(f=arguments.mn):
    
    checkpoint = torch.load(f=arguments.mn,map_location=torch.device('cpu'))
    
    if checkpoint['model_name'] == 'vgg':
        model = models.vgg16(pretrained=True)
        
    elif checkpoint['model_name'] == 'alexnet':  
        model = models.alexnet(pretrained=True)
    else:
        print("Architecture not recognized.")
        
    for param in model.parameters():
            param.requires_grad = False    
    
    model.class_to_idx = checkpoint['class_to_idx']

    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(checkpoint['clf_input'], checkpoint['hidden_layer_units'])),
                                        ('relu', nn.ReLU()),
                                        ('drop', nn.Dropout(arguments.p)),
                                        ('fc2', nn.Linear(checkpoint['hidden_layer_units'], 5)),
                                        ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

## Load the image and check if Dicom or Nifti
def check_filetype(im = arguments.im):
    
    file_ext_name = os.path.splitext(arguments.im)[0]
    file_ext = os.path.splitext(arguments.im)[1]
    
    if file_ext == '.jpg':
        filename = file_ext_name + file_ext
        with open(arguments.im, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB'), filename
        #filename = file_ext_name + file_ext
        #return img #,filename

    elif file_ext == '.dcm' or '.DCM':
            img_data = dicom2jpg.dicom2img(im)
            img = Image.fromarray(img_data)
            img.save(file_ext_name + '.jpg')
            filename = file_ext_name + '.jpg'        
            return img,filename
    
    ''' #To be implemented in the future
    else: 
        if file_ext == '.nii':
            img_data = nib.load(im)
            im_shape = nii_img.shape
            nib.save(img_data[((im_shape[2] -1)/2)])
            return img
    '''    

img,filename = check_filetype(im = arguments.im)

model = load_checkpoint(img)
#print(model)

def pil_loader(img) :
    return img.convert('RGB') 
#print(img.size)



def pre_image(img,mn=arguments.mn):
   
   #img = Image.open(image_path)
   img =pil_loader(img)
   #img.convert("L")


   mean = [0.485, 0.456, 0.406] 
   std = [0.229, 0.224, 0.225]

   transform_norm = transforms.Compose([transforms.ToTensor(), 
   transforms.Resize((224,224)),transforms.Normalize(mean, std)])
  
   # get normalized image
   img_normalized = transform_norm(img).float()
   img_normalized = img_normalized.unsqueeze_(0)
   
   # input = Variable(image_tensor)
   img_normalized = img_normalized.to('cpu')
   #print(img_normalized.size)
   # 
   classes = ['DSC','FLAIR','T1','T1ce','T2']  
   with torch.no_grad():
      model.eval()  
      output =model(img_normalized)
      #print(output)
      index = output.data.cpu().numpy().argmax()    
      #print(index)
      class_name = classes[index]
      return class_name

## Do the inferencing

print('\n************************************************\n')
predict_class = pre_image(img,model)
#print('     Filename is {}'.format(filename))
print('     The predicted class is: {}'.format(predict_class))
end = time.time()
total_time = end - start
print("\n     Total time taken for inference in seconds: "+ str(total_time))
print('\n************************************************')

## Cleaning

file_ext_name = os.path.splitext(arguments.im)[0]
file_ext = os.path.splitext(arguments.im)[1]

if file_ext =='.jpg':
    pass
else:
    os.remove(filename)

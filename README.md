# BrainMRI_PulseSeqType_Classifier
Classifies DICOM and JPG images into DSC, FLAIR, T1, T1ce, T2 classes. Can be used where the information regarding the images has been stripped off.

USAGE:
1) Download everything in one directory
2) Install libraries : pip install -r requirements.txt
3) Cd into the directory and type (for testing on cpu):

python test.py --im=<image_name>


4) Default setting run with command : 
python test.py 

    -image is '_FLAIR_t2_FLAIR_spc_Ulleval_0002.jpg'
    -model is 'CraiClassifier_v1.pth'


5) Results fromm experiment are in file ResultsFromExp_05012023.txt

6) Examples:


TEST DICOM IMAGES:
=================

python test.py --im=1_FLAIR.dcm

python test.py --im=t2_1.dcm


TEST JPG IMAGES:
==============

python test.py --im=_DSC_Sailor_0001.jpg 

python test.py --im=_FLAIR_t2_FLAIR_spc_Ulleval_0002.jpg

python test.py --im=_T1_pre_Sailor_0008.jpg

python test.py --im=T1cm_image_hgg_0007.jpg

For changing the classifier from default (CraiClassifier_v1.pth) to CraiClassifier_v2.pth:

python test.py --im=_t2_tse_tra_Ulleval_0063.jpg --mn=CraiClassifier_v2.pth

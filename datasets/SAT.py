import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image

from torchvision import transforms
from torch.utils.data.dataset import Dataset

imgSize = 256

class Satellite_Dataset(Dataset):
    def __init__(self, imagenames, split="train"):

        if split == "train":
            self.root_dir = "sat_data_final\\train"
        elif split == "val":
            self.root_dir = "sat_data_final\\val"
        elif split == "test":
            self.root_dir = "sat_data_final\\test"

        self.normals_dir = self.root_dir + "\\normal"
        self.anomalies_dir = self.root_dir + "\\anomalous"

        self.imagenames = imagenames
        self.transformations = transforms.Compose([
            transforms.Resize((imgSize,imgSize)),
            transforms.ToTensor(),
            ])
        
    def __getitem__(self, idx):
        
        img = Image.open(os.path.join(self.normals_dir, self.imagenames[idx])) if pathlib.Path(os.path.join(self.normals_dir, self.imagenames[idx])).exists() else Image.open(os.path.join(self.anomalies_dir, self.imagenames[idx]))
        
        if self.transformations != None:
            img = self.transformations(img)
        return img
    
    def __len__(self): 
        return len(self.imagenames)
      
      
def get_satellite_data():
    train_normals = os.listdir("sat_data_final\\train\\normal")
    val_normals = os.listdir("sat_data_final\\val\\normal")
    test_normals = os.listdir("sat_data_final\\test\\normal")
    val_anomalies = os.listdir("sat_data_final\\val\\anomalous")
    test_anomalies = os.listdir("sat_data_final\\test\\anomalous")
    
    return train_normals, val_normals, test_normals, val_anomalies, test_anomalies

# class SIIM_Dataset(Dataset):
#     def __init__(self, traindir, imagenames, labels):
#         self.traindir = traindir
#         self.imagenames = imagenames
#         self.labels = labels
#         self.transformations = transforms.Compose([
#                                      transforms.Resize((imgSize,imgSize)),
#                                      transforms.ToTensor()
#                                     ])
#     def __getitem__(self, idx):
#         img = Image.open(os.path.join(self.traindir, self.imagenames[idx]))
#         if self.transformations != None:
#             img = self.transformations(img)
#         return img, self.labels[idx]
    
#     def __len__(self): 
#         return len(self.imagenames)


    
# def get_siim_data():
#     rootdir = "./data/SIIM/jpeg/"
#     traindir = rootdir+"train"
#     testdir = rootdir+"test"
#     train_images = os.listdir(traindir)
#     test_images = os.listdir(testdir)
#     train_csv = pd.read_csv('./data/SIIM/train.csv')[1:].reset_index(drop=True)
#     benign = train_csv.loc[train_csv["benign_malignant"]=="benign"]
#     malignant = train_csv.loc[train_csv["benign_malignant"]=="malignant"]
    
#     benigns = []
#     for i in range(0, len(benign["image_name"])):
#         benigns.append(benign["image_name"].iloc[i]+".jpg")
        
#     malignants = []
#     for i in range(0, len(malignant["image_name"])):
#         malignants.append(malignant["image_name"].iloc[i]+".jpg")    
    
#     return benigns, malignants

import os
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
import csv
import pickle




class CamelyonFeatures(Dataset):

    def open_hdf5(self):
        self.dataset = h5py.File(self.data_dir, 'r')

    def select_slides(self):
        h5_data = h5py.File(self.data_dir, 'r')
        self.slide_names = list(h5_data.keys())
        self.data_len = len(self.slide_names)
        h5_data.close()

    def __init__(self, conf, train=True):

        self.tasks = conf.tasks

        filename = conf.train_fname if train else conf.test_fname
        self.data_dir = os.path.join(conf.data_dir, filename)

        self.select_slides()
    
    def __len__(self):
        return self.data_len

    def __getitem__(self, i):
        
        if not hasattr(self, 'dataset'):
            self.open_hdf5()

        slide_name = self.slide_names[i]

        slide = self.dataset[slide_name]
        patches = slide['img'][:]
        label = slide.attrs['label']

        data_dict = {'input': patches}
        for task in self.tasks.values():
            data_dict[task['name']] = label

        return data_dict

class CamelyonFeatures0(Dataset):



    def __init__(self, conf, train=True):

        self.tasks = conf.tasks
        self.type=train
        self.slides = []
        self.labels = []
        with open('/home/ubuntu/qr/CAMELYON16_r50/label.csv', 'r') as file:
            reader = csv.reader(file)
            count=0
            for row in reader:
                if ("patient" not in row[0]):
                    count+=1
                    if train:
                        if("test" not in row[0]):
                            self.slides.append('/home/ubuntu/qr/CAMELYON16_r50/pt/'+row[0]+".pt")
                            self.labels.append(int(row[1]))
                    else:
                        if("test" in row[0]):
                            self.slides.append('/home/ubuntu/qr/CAMELYON16_r50/pt/'+row[0]+".pt")
                            self.labels.append(int(row[1]))
            print(count)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        patches = self.slides[i]
        patches = torch.load(patches)
        #print(patches.shape)       
        if(self.type==True):
            temp_list=[]
            for i1 in range(patches.shape[0]):
                temp_list.append(i1)
            ids = random.sample(temp_list, int(patches.shape[0]*0.74))
            patches=patches[ids]
        #print(patches.shape)
        #print()
        label = self.labels[i]
        data_dict = {'input': patches}
        for task in self.tasks.values():
            data_dict[task['name']] = label
        return data_dict
    
    
class CamelyonFeatures1(Dataset):



    def __init__(self, conf, train=True):

        self.tasks = conf.tasks
        self.type=train
        self.slides = []
        self.labels = []
        with open('/home/ubuntu/qr/CAMELYON16_r50/label.csv', 'r') as file:
            reader = csv.reader(file)
            temp_list=[]
            for i1 in range(398):
                temp_list.append(i1)
            ids = random.sample(temp_list, len(temp_list))
            count=0
            for row in reader:
                
                if ("patient" not in row[0]):
                    count+=1
                    if train:
                        if(count<=132 or count>=265):
                        #if(count>=133):
                        #if(count<266):
                            self.slides.append('/home/ubuntu/qr/CAMELYON16_r50/pt/'+row[0]+".pt")
                            self.labels.append(int(row[1]))
                    else:
                        if(132<count<265):
                        #if(count<133):
                        #if(count>=266):
                            self.slides.append('/home/ubuntu/qr/CAMELYON16_r50/pt/'+row[0]+".pt")
                            self.labels.append(int(row[1]))

            print(count)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        patches = self.slides[i]
        patches = torch.load(patches)
        #print(patches.shape)       
        if(self.type==True):
            temp_list=[]
            for i1 in range(patches.shape[0]):
                temp_list.append(i1)
            ids = random.sample(temp_list, int(patches.shape[0]*0.9))
            patches=patches[ids]
        #print(patches.shape)
        #print()
        label = self.labels[i]
        data_dict = {'input': patches}
        for task in self.tasks.values():
            data_dict[task['name']] = label
        return data_dict
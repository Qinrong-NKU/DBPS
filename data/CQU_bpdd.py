import os
from os import path
from torchvision import transforms,utils
from torchvision import  datasets
import torch
print(torch.__version__)

import torch.utils.data
import numpy as np
from torchvision.datasets import ImageFolder
import torch
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torch.nn.functional as F

class MyDataset(ImageFolder):
    IMG_SIZE = (1200, 900)
    def __init__(self, conf,_type='train'):
        
        self.patch_size = conf.patch_size
        self.patch_stride = conf.patch_stride
        self.tasks = conf.tasks

        if _type == 'train':
            data_dir=path.join(conf.data_dir,'train')
        elif _type == 'val':
            data_dir=path.join(conf.data_dir,'val')
        elif _type == 'test':
            data_dir=path.join(conf.data_dir,'test')
        
        
        
        transform_list = [
            transforms.Resize([*self.IMG_SIZE])
        ]

        if _type == 'train':
            transform_list += [
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                transforms.RandomAffine(degrees=0, translate=(100 / self.IMG_SIZE[1], 100 / self.IMG_SIZE[0])),
            ]
        
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
           
        self.transform = transforms.Compose(transform_list)
        
        super(MyDataset, self).__init__(data_dir, transform=self.transform )
      
    def __getitem__(self, index):
        # get the image and label from the parent class
        path, category = self.imgs[index]
        img = self.loader(path)
        img = self.transform(img)
        # add your own processing code here
        # for example, you can convert the image to a tensor
        #image = torch.from_numpy(image)

        patch_size = self.patch_size
        patch_stride = self.patch_stride


        # Extract patches
        patches = img.unfold(
            1, patch_size[0], patch_stride[0]
        ).unfold(
            2, patch_size[1], patch_stride[1]
        ).permute(1, 2, 0, 3, 4)

        patches = patches.reshape(-1, *patches.shape[2:])

        data_dict = {'input': patches}
        for task in self.tasks.values():
            data_dict[task['name']] = category

        return data_dict

class MyDataset1(ImageFolder):
    IMG_SIZE = (1200, 900)

    def __init__(self, conf, _type='train'):

        self.patch_size = conf.patch_size
        self.patch_stride = conf.patch_stride
        self.tasks = conf.tasks
        self.patches={}
        self.type=_type

        if _type == 'train':
            data_dir = path.join(conf.data_dir, 'train')
        elif _type == 'val':
            data_dir = path.join(conf.data_dir, 'val')
        elif _type == 'test':
            data_dir = path.join(conf.data_dir, 'test')

        transform_list = [
            transforms.Resize([*self.IMG_SIZE])
        ]

        if _type == 'train':
            transform_list += [
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                transforms.RandomAffine(degrees=0, translate=(100 / self.IMG_SIZE[1], 100 / self.IMG_SIZE[0])),
            ]

        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        # if _type == 'train':
        #     transform_list += [
        #         transforms.RandomErasing()
        #     ]

        self.transform = transforms.Compose(transform_list)

        super(MyDataset1, self).__init__(data_dir, transform=self.transform)

    def __getitem__(self, index):
        # get the image and label from the parent class
        path, category = self.imgs[index]
        if(index not in self.patches.keys()):
            img = self.loader(path)
            img = self.transform(img)
            # add your own processing code here
            # for example, you can convert the image to a tensor
            # image = torch.from_numpy(image)

            patch_size = self.patch_size
            patch_stride = self.patch_stride

            # Extract patches
            patches = img.unfold(
                1, patch_size[0], patch_stride[0]
            ).unfold(
                2, patch_size[1], patch_stride[1]
            ).permute(1, 2, 0, 3, 4)

            patches = patches.reshape(-1, *patches.shape[2:])
            if(self.type=="test"):
                self.patches[index]=patches.numpy()
        else:
            patches=torch.tensor(self.patches[index])

        data_dict = {'input': patches}
        for task in self.tasks.values():
            data_dict[task['name']] = category

        return data_dict

class MyDataset2(ImageFolder):
    IMG_SIZE = (1200, 900)

    def __init__(self, conf, _type='train'):

        self.patch_size = conf.patch_size
        self.patch_stride = conf.patch_stride
        self.tasks = conf.tasks
        self.patches = {}
        self.type = _type

        if _type == 'train':
            data_dir = path.join(conf.data_dir, 'train')
        elif _type == 'val':
            data_dir = path.join(conf.data_dir, 'val')
        elif _type == 'test':
            data_dir = path.join(conf.data_dir, 'test')

        transform_list = [
            transforms.Resize([*self.IMG_SIZE])
        ]

        if _type == 'train':
            transform_list += [
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                transforms.RandomAffine(degrees=0, translate=(100 / self.IMG_SIZE[1], 100 / self.IMG_SIZE[0])),
            ]

        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        if _type == 'train':
            transform_list += [
                transforms.RandomErasing(scale=(0.5, 0.5),p=1)
            ]

        self.transform = transforms.Compose(transform_list)

        super(MyDataset2, self).__init__(data_dir, transform=self.transform)

    def __getitem__(self, index):
        # get the image and label from the parent class
        path, category = self.imgs[index]
        if (index not in self.patches.keys()):
            img = self.loader(path)
            img = self.transform(img)
            # add your own processing code here
            # for example, you can convert the image to a tensor
            # image = torch.from_numpy(image)

            patch_size = self.patch_size
            patch_stride = self.patch_stride

            # Extract patches
            patches = img.unfold(
                1, patch_size[0], patch_stride[0]
            ).unfold(
                2, patch_size[1], patch_stride[1]
            ).permute(1, 2, 0, 3, 4)

            patches = patches.reshape(-1, *patches.shape[2:])
            if (self.type == "test"):
                pass
                # self.patches[index] = patches.numpy()
        else:
            patches = torch.tensor(self.patches[index])

        data_dict = {'input': patches}
        for task in self.tasks.values():
            data_dict[task['name']] = category

        return data_dict

class MyDataset3(ImageFolder):
    # IMG_SIZE = (1000, 1000)
    IMG_SIZE = (1200, 900)

    def __init__(self, conf, _type='train'):

        self.patch_size = conf.patch_size
        self.patch_stride = conf.patch_stride
        self.tasks = conf.tasks
        self.patches = {}
        self.type = _type
        self.N=conf.N
        self.I=conf.I
        self.M=conf.M

        if _type == 'train':
            data_dir = path.join(conf.data_dir, 'train')
        elif _type == 'val':
            data_dir = path.join(conf.data_dir, 'val')
        elif _type == 'test':
            data_dir = path.join(conf.data_dir, 'test')

        transform_list = [
            transforms.Resize([*self.IMG_SIZE])
        ]

        if _type == 'train':
            transform_list += [
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                transforms.RandomAffine(degrees=0, translate=(100 / self.IMG_SIZE[1], 100 / self.IMG_SIZE[0])),
            ]

        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        # if _type == 'train':
        #     transform_list += [
        #         transforms.RandomErasing(scale=(0.1, 0.1),p=1)
        #     ]
        
        # if(self.type=="train"):
        #     transform_list1 = [
        #         transforms.RandomErasing(scale=(0.94,0.94),p=1,ratio=(1, 1))
        #     ]
        #     self.transform1 = transforms.Compose(transform_list1)
        

        self.transform = transforms.Compose(transform_list)
        self.list=[]
        for i in range(conf.N):
            self.list.append(i)

        super(MyDataset3, self).__init__(data_dir, transform=self.transform)
        print(len(self.imgs))

    def __getitem__(self, index):
        # get the image and label from the parent class
        path, category = self.imgs[index]
        if (index not in self.patches.keys()):
            img = self.loader(path)
            img = self.transform(img)
            # add your own processing code here
            # for example, you can convert the image to a tensor
            # image = torch.from_numpy(image)

            patch_size = self.patch_size
            patch_stride = self.patch_stride

            # Extract patches
            patches = img.unfold(
                1, patch_size[0], patch_stride[0]
            ).unfold(
                2, patch_size[1], patch_stride[1]
            ).permute(1, 2, 0, 3, 4)

            patches = patches.reshape(-1, *patches.shape[2:])
            #if (self.type == "test" and index<4000):
                #np.save("npy/"+str(index), patches.numpy())
                #self.patches[index] = patches.numpy()
        else:
            pass
            #patches = torch.tensor(np.load("npy/"+str(index)+".npy"))
            #patches = torch.tensor(self.patches[index])
            

        if (self.type == "train" ):
            ids=random.sample(self.list,self.M+int((self.N-self.M)*1.0))
            patches=patches[ids]
        # print(patches.shape)

        data_dict = {'input': patches}
        for task in self.tasks.values():
            data_dict[task['name']] = category

        return data_dict

    '''
    @property
    def class_frequencies(self):
        """Compute and return the class specific frequencies."""
        freqs = np.zeros(len(self.CLASSES), dtype=np.float32)
        for image, category in self._data:
            freqs[category] += 1
        return freqs / len(self._data)
    '''

    

    

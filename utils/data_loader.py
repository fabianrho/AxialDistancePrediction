import torch
from torchvision.transforms import v2 as transforms
from torchvision.io import read_image
import os
import glob
import matplotlib.pyplot as plt



class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path):
        super(ImageDataset, self).__init__()
        
        if isinstance(folder_path, list):
            self.img_files = []
            for folder in folder_path:
                self.img_files += glob.glob(os.path.join(folder,'Image/class_0','*.png'))
        else:
            self.img_files = glob.glob(os.path.join(folder_path,'Image/class_0','*.png'))
        # adjust these paths according to your folder structure

            

    def __getitem__(self, index):
        img_path = self.img_files[index]
        data = read_image(img_path)[0,:,:].unsqueeze(0)

        return data.float()/255
    
    def __len__(self):
        return len(self.img_files)
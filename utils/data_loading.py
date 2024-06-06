import torch
# from torchvision import transforms, datasets
from torchvision.transforms import v2 as transforms
from torchvision.io import read_image
import torchvision
import os
import glob
import cv2
import time
import random
import matplotlib.pyplot as plt

# import torch.utils.data as data

class BarlowTwinsDataset(torch.utils.data.Dataset):
    #https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    def __init__(self, folder_path, create_three_channels=False, resize_to = None, use_crop = False):
        super(BarlowTwinsDataset, self).__init__()
        
        if isinstance(folder_path, list):
            self.img_files = []
            for folder in folder_path:
                self.img_files += glob.glob(os.path.join(folder,'Image/class_0','*.png'))
        else:
            self.img_files = glob.glob(os.path.join(folder_path,'Image/class_0','*.png'))

        self.transform = transforms.Compose([
            # transforms.RandomResizedCrop(192, interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            # transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=19)], p=1),
            transforms.RandomSolarize(0.5, p=0),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            # transforms.RandomResizedCrop(192, interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            # transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=19)], p=0.1),
            transforms.RandomSolarize(0.5, p=0.2),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])

        self.create_three_channels = create_three_channels

        self.resize_to = resize_to
        if self.resize_to is not None:
            if use_crop == False:
                self.resize = transforms.Resize(resize_to)

            else:
                self.resize = transforms.CenterCrop(resize_to)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        data = read_image(img_path)[0,:,:].unsqueeze(0)

        if self.resize_to is not None:
                data = self.resize(data)

        view1 = self.transform(data)
        view2 = self.transform_prime(data)

        # copy channel to match the 3 channel input
        if self.create_three_channels:
            view1 = torch.cat([view1, view1, view1], dim=0)
            view2 = torch.cat([view2, view2, view2], dim=0)


        return view1.float()/255, view2.float()/255
    
    def __len__(self):
        return len(self.img_files)

class RotationDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path):
        super(RotationDataset, self).__init__()
        
        if isinstance(folder_path, list):
            self.img_files = []
            for folder in folder_path:
                self.img_files += glob.glob(os.path.join(folder,'Image/class_0','*.png'))
        else:
            self.img_files = glob.glob(os.path.join(folder_path,'Image/class_0','*.png'))

        # self.transform = transforms.Compose([
        #     transforms.RandomRotation(180),
        # ])

            

    def __getitem__(self, index):
        img_path = self.img_files[index]
        data = read_image(img_path)[0,:,:].unsqueeze(0)
        # get random angel
        angle = random.randint(-120, 120)
        rotated_image = torchvision.transforms.functional.rotate(data, angle)
        # print(rotated_image.shape)
        # resize
        # rotated_image = torchvision.transforms.functional.resize(rotated_image, (192, 192))



        return rotated_image.float()/255, angle
    
    def __len__(self):
        return len(self.img_files)



class DataLoaderSegmentation(torch.utils.data.Dataset):
    def __init__(self, folder_path, augmentation=False, n_augmentations=1, flip_and_rotate_aug = False, create_three_channels=False, resize_to=None, use_crop = False):
        super(DataLoaderSegmentation, self).__init__()
        if isinstance(folder_path, list):
            self.img_files = []
            for folder in folder_path:
                self.img_files += glob.glob(os.path.join(folder,'Image/class_0','*.png'))
            self.mask_files = []
            for folder in folder_path:
                mask_files = glob.glob(os.path.join(folder,'Label/class_0','*.png'))
                self.mask_files += mask_files
        else:
            self.img_files = glob.glob(os.path.join(folder_path,'Image/class_0','*.png'))

            self.mask_files = []
            for img_path in self.img_files:
                self.mask_files.append(os.path.join(folder_path,'Label/class_0',os.path.basename(img_path)))

        if augmentation:
            self.img_files = self.img_files * n_augmentations
            self.mask_files = self.mask_files * n_augmentations

        self.augmentation = augmentation
        self.create_three_channels = create_three_channels

        self.resize_to = resize_to
        if self.resize_to is not None:
            if use_crop == False:
                self.resize = transforms.Resize(resize_to)

            else:
                self.resize = transforms.CenterCrop(resize_to)

        #### flip and rotation
        if flip_and_rotate_aug:
            self.augmenter = torch.nn.Sequential(
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=30, interpolation=transforms.InterpolationMode.BILINEAR),
            )
        else:
            ### random affine
            self.augmenter = torch.nn.Sequential(
                transforms.RandomAffine(degrees=45, translate=(0.2, 0.2), scale=(0.8, 1.4), shear=10, interpolation=transforms.InterpolationMode.BILINEAR, fill=0),
            )
                

        # #### RandAugment
        # self.augmenter = torch.nn.Sequential(
        #     transforms.RandAugment(2, 9, interpolation=transforms.InterpolationMode.BILINEAR),
        # )        

        


    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]

            data = read_image(img_path)[0,:,:].unsqueeze(0)
            label = read_image(mask_path)[0,:,:].unsqueeze(0)



            if self.resize_to is not None and data.shape[1] != self.resize_to:
                data = self.resize(data)
                label = self.resize(label)


            if self.augmentation:
                # concatenate data and label in order to apply the same augmentation
                data_and_label = torch.cat([data, label, label], dim=0) 
                

                data_and_label = self.augmenter(data_and_label)

                # separate data and label
                data = data_and_label[0,:,:].unsqueeze(0)
                label = data_and_label[1,:,:].unsqueeze(0)


            # normalize
            data = data.float()/255
            label = label.float()/255
            label[label > 0.5] = 1.
            label[label <= 0.5] = 0.
            

            if self.create_three_channels:
                data = torch.cat([data, data, data], dim=0)

            return data.float(), label.float()

    def __len__(self):
        return len(self.img_files)
    

if __name__ == "__main__":

    # data_dir = "../internship/192/data_192_limited/10_percent/Lung/axial/Preprocessed_data/train"
    data_dir = "../internship/192/data_192/Pelvis/axial/Preprocessed_data/train"


    train_data = BarlowTwinsDataset(data_dir)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=False, num_workers=4, drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    print(device)
    i = 0
    import numpy as np


    for batch in range(2):
        for data, _ in train_loader:
            image = data[0].numpy()
            print(image.shape)


            # image = image[:,:].squeeze()

            # image = np.stack([image, image, image], axis=-1)
            # transpose
            # image = np.transpose(image, (1, 2, 0))

            # plt.imsave(image)
            # plt.imsave(f"temp/data{i}.png", image, cmap = 'gray')
            # plt.imsave(f"temp/label{i}.png", label, cmap = 'gray')
            i += 1
            break
        


        

        



import torch
import torch.utils.data as data
import nibabel as nib
import os
import random 
from tqdm import tqdm
import numpy as np
import pickle
import time

from torchvision.transforms import v2 as transforms
from torchvision.io import read_image

from lightly.transforms.simclr_transform import SimCLRTransform


class AxialDistanceDataset(data.Dataset):
    """Creates dataset for axial distance prediction, returning two images and the normalized
    distance between the two slices in the volume.
    """

    def __init__(self, data_dir, nifti_dir, validation = False, patient_dict_path = None, max_patients = None, pairs_per_patient = 200, max_distance_cm = None, transform=None, resize_to=(256,256), create_three_channels=False):
        self.data_dir = data_dir
        self.nifti_dir = nifti_dir
        self.validation = validation
        self.pairs_per_patient = pairs_per_patient
        # self.nifit_filesnames = [f.name for f in os.scandir(nifti_dir) if f.name.endswith('.nii') or f.name.endswith('.nii.gz')]
        self.transform = transform
        self.create_three_channels = create_three_channels
        self.resize_to = resize_to
        # if self.resize_to is not None:
        #     self.resize = transforms.Resize(resize_to)

        # create nested list with all the images
        if ("Rib" in data_dir or "Pelvis" in data_dir) and ("full" not in data_dir and "internship" in data_dir):
            self.patient_ids = list(set([f.name.split('_')[2] for f in os.scandir(self.data_dir)]))
        elif ("pancreas" in data_dir or "Pelvis" in data_dir or "Rib" in data_dir):# and ("full" in data_dir):
            self.patient_ids = list(set([f.name.split('_')[0] for f in os.scandir(self.data_dir)]))
        self.patient_ids.sort()
        self.patient_ids = self.patient_ids[:max_patients]

        # preload the image files per patient
        if patient_dict_path is None:
            self.patient_files = {}
            for id in tqdm(self.patient_ids):
                if ("Rib" in data_dir or "Pelvis" in data_dir) and ("full" not in data_dir and "internship" in data_dir):
                    self.patient_files[id] = [f.name for f in os.scandir(self.data_dir) if f.name.split('_')[2] == id]
                elif ("pancreas" in data_dir or "Pelvis" in data_dir or "Rib" in data_dir):# and ("full" in data_dir):
                    self.patient_files[id] = [f.name for f in os.scandir(self.data_dir) if f.name.split('_')[0] == id]
        else:
            # reload dictionary
            with open(patient_dict_path, 'rb') as handle:
                self.patient_files = pickle.load(handle)


        self.z_spacings = {}
        self.scan_height = {}
        for id in self.patient_ids:
            if "Rib" in data_dir:
                img = nib.load(os.path.join(self.nifti_dir, f'{id}-image.nii.gz'))
            elif "pancreas" in data_dir:
                img = nib.load(os.path.join(self.nifti_dir, f'{id}.nii.gz'))
            elif "Pelvis" in data_dir:
                img = nib.load(os.path.join(self.nifti_dir, f'CTPelvic1K_{id}_0000.nii.gz'))
            self.z_spacings[id] = img.header.get_zooms()[2]
            self.scan_height[id] = img.shape[2] * self.z_spacings[id]
        
        # get the maximum height across the all the scans
        if max_distance_cm is None:
            self.max_height = max(self.scan_height.values())
        else:
            self.max_height = max_distance_cm*10 # to mm






        
        # all possible pairs per patient 
        self.pairs = []
        for id in self.patient_ids:
            patient_images = self.patient_files[id]
            patient_slices = sorted([int(f.split('_')[-1].split(".")[0]) for f in patient_images])
            current_patient_pairs = []
            for i in range(len(patient_slices)):
                for j in range(i+1, len(patient_slices)):
                    slice1 = f"{id}_{patient_slices[i]}.png"
                    slice2 = f"{id}_{patient_slices[j]}.png"
                    
                    current_patient_pairs.append((slice1, slice2))
            # limit current_patient_pairs to max_distance_cm
            if max_distance_cm is not None:
                max_distance_mm = max_distance_cm*10
                if "lesion" in self.patient_files[id][0]:
                    current_patient_pairs = [pair for pair in current_patient_pairs if abs(int(pair[0].split('_')[-1].split(".")[0]) - int(pair[1].split('_')[-1].split(".")[0])) * self.z_spacings[id] <= max_distance_mm]
                else:
                    current_patient_pairs = [pair for pair in current_patient_pairs if abs(int(pair[0].split('_')[1].split(".")[0]) - int(pair[1].split('_')[1].split(".")[0])) * self.z_spacings[id] <= max_distance_mm]

            if self.pairs_per_patient == "all":
                self.pairs.extend(current_patient_pairs)
            else:
                if len(current_patient_pairs) < self.pairs_per_patient:
                    self.pairs.extend(current_patient_pairs)
                else:
                    # get same pairs for each patient, for reproducibility
                    #### THIS IS NEW, COMPARED TO FIRST SUBMISSION ####
                    random.seed(int(''.join(str(ord(c)) for c in id)))
                    #### ####
                    self.pairs.extend(random.sample(current_patient_pairs, self.pairs_per_patient))


        # else:
        #     # repeat the patient ids by number of pairs per patient
        #     self.patient_ids = self.patient_ids * self.pairs_per_patient


        # augmentations
        self.augmenter = torch.nn.Sequential(
            transforms.RandomRotation(degrees=90, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop(size=self.resize_to, scale=(0.15, 0.5), ratio=(1, 1), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        )


    def __len__(self):
        return len(self.pairs)




    def __getitem__(self, index):
        # if self.pairs_per_patient == "all":

        slice1, slice2 = self.pairs[index]
        patient_id = slice1.split('_')[0]
        patient_images = self.patient_files[patient_id]


        data1 = read_image(os.path.join(self.data_dir, slice1))[0,:,:].unsqueeze(0)
        data2 = read_image(os.path.join(self.data_dir, slice2))[0,:,:].unsqueeze(0)

        # get the distance between the two slices in mm making use of the z-spacing in the header data of the nifti file
        if "lesion" in patient_images[0]:
            distance_mm = abs(int(slice2.split('_')[-1].split(".")[0]) - int(slice1.split('_')[-1].split(".")[0])) * self.z_spacings[patient_id]
        else:
            distance_mm = abs(int(slice2.split('_')[1].split(".")[0]) - int(slice1.split('_')[1].split(".")[0])) * self.z_spacings[patient_id] 


        # create three channels, needed if pretrained model is trained on RGB images (like imagenet pretrained swin-unet)
        if self.create_three_channels:
            data1 = torch.cat([data1, data1, data1], dim=0)
            data2 = torch.cat([data2, data2, data2], dim=0)


        slice1_id = slice1.split('_')[-1].split(".")[0]
        slice2_id = slice2.split('_')[-1].split(".")[0]

        
        #### THIS IS NEW, COMPARED TO FIRST SUBMISSION ####
        if self.validation:
            # set seed to obtain same augmentations for validation set (reproducibility)
            seed1 = int(patient_id.replace("RibFrac", "") + slice1_id)
            random.seed(seed1)
            torch.manual_seed(seed1)
        #### ####
        data1 = self.augmenter(data1)
        

        #### THIS IS NEW, COMPARED TO FIRST SUBMISSION
        if self.validation:
            # set seed to obtain same augmentations for validation set (reproducibility)
            seed2 = int(patient_id.replace("RibFrac", "") + slice2_id)
            random.seed(seed2)
            torch.manual_seed(seed2)
        #### ####

        data2 = self.augmenter(data2)

        # normalize the images
        data1 = data1.float()/255
        data2 = data2.float()/255

        
        # return the images and the normalized distance between the two slices (between 0 and 1)
        return data1, data2, distance_mm/self.max_height


if __name__ == "__main__":

    pass
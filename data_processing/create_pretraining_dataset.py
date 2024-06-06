import torch
import nibabel as nib
import numpy as np
import os
from matplotlib import image
from tqdm import tqdm


def generate_dataset(Path_to_CT, saving_folder):

    
    # saving_folder = os.path.join(saving_folder)

    for folder in tqdm(os.listdir(Path_to_CT)):
        if "PETCT" in folder:

            for i, sub_folder in enumerate(os.listdir(os.path.join(Path_to_CT, folder))):
                ct_file = os.path.join(Path_to_CT, folder, sub_folder, "CT.nii.gz")
                print(folder)

                patient_id = folder.split("_")[1]
                if len(os.listdir(os.path.join(Path_to_CT, folder))) > 1:
                    patient_id = f"{patient_id}_{i}"

                original_data  = nib.load(ct_file).get_fdata()




                # normalize orginal data per slice
                for i in range(original_data.shape[2]):
                    original_data[:,:,i] = (original_data[:,:,i] - np.min(original_data[:,:,i])) / (np.max(original_data[:,:,i]) - np.min(original_data[:,:,i]))

                

                for i in range(original_data.shape[2]):
                    original_slice = original_data[:,:,i]
                    if os.path.join(saving_folder, f"{patient_id}_{i}.png") not in os.listdir(saving_folder):
                        image.imsave(os.path.join(saving_folder, f"{patient_id}_{i}.png"), original_slice, cmap="gray", vmax=1, vmin=0)

        # break
    return None

if __name__ == "__main__":
    generate_dataset("/home/llong/PET_CT", "data/PET_CT")
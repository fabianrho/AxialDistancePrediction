import torch
import nibabel as nib
import numpy as np
import os
from matplotlib import image
from tqdm import tqdm






def generate_dataset(Path_to_CT, Path_to_Annotation, saving_folder, use_label_contrain = True):

    Set_up_dir(saving_folder)
    saving_folder = os.path.join(saving_folder, "Preprocessed_data")

    for file in tqdm(os.listdir(Path_to_CT)):

        if "CTPelvic1K" in file:
            ct_file = os.path.join(Path_to_CT, file)

            patient_id = file.split("_")[1]
            annotation_file = os.path.join(Path_to_Annotation, f"CTPelvic1K_{patient_id}.nii.gz")

            original_data  = nib.load(ct_file).get_fdata()
            if use_label_contrain:

                label_data  = nib.load(annotation_file).get_fdata()

        elif "RibFrac" in file:
            ct_file = os.path.join(Path_to_CT, file)

            patient_id = file.split("-")[0]

            print(patient_id)

            if patient_id not in ["RibFrac452", "RibFrac485", "RibFrac490"]:
                continue
            annotation_file = os.path.join(Path_to_Annotation, f"{patient_id}-rib-seg.nii.gz")

            original_data  = nib.load(ct_file).get_fdata()
            if use_label_contrain:
                label_data  = nib.load(annotation_file).get_fdata()


        else:
            patient_id = file.split(".")[0]
            ct_file = os.path.join(Path_to_CT, file)
            annotation_file = os.path.join(Path_to_Annotation, f"label{patient_id}.nii.gz")
            original_data  = nib.load(ct_file).get_fdata()

            if use_label_contrain:
                label_data  = nib.load(annotation_file).get_fdata()

        # if patient_id not in [452,485,490]:
        #     continue


        # normalize orginal data per slice
        for i in range(original_data.shape[2]):
            original_data[:,:,i] = (original_data[:,:,i] - np.min(original_data[:,:,i])) / (np.max(original_data[:,:,i]) - np.min(original_data[:,:,i]))

            # remove slice
        
            # print(np.min(original_data[:,:,i]), np.max(original_data[:,:,i]))
            # print(original_data[:,:,i])

        for i in range(original_data.shape[2]):
            original_slice = original_data[:,:,i]

            if use_label_contrain:
                label_slice = label_data[:,:,i]
            else: 
                label_slice = np.ones_like(original_slice)


            if np.sum(label_slice) > 0:
                try:
                    image.imsave(os.path.join(saving_folder, "train/Image/class_0", f"{patient_id}_{i}.png"), original_slice, cmap="gray", vmax=1, vmin=0)
                    if use_label_contrain:
                        image.imsave(os.path.join(saving_folder, "train/Label/class_0", f"{patient_id}_{i}.png"), label_slice, cmap="gray", vmax=1, vmin=0)
                except:
                    print(f"Error in {patient_id}_{i}.png")
                    pass

        # break
    return None


def Set_up_dir(Path_to_data_folder):
        """
        This function creates folders that are used to store the training test and validation data
    
        Parameters
        ----------
        Path_to_data_folder : string
            Path to the folder where you want to store the preprocessing results..
    
        Returns
        -------
        Dictionary_paths : dict
            Dictionary with all of the paths that are used during pre-processing or training.
    
        """
        
        Main_Data_folder=Path_to_data_folder
        Data_folder=os.path.join(Main_Data_folder,"Preprocessed_data")
    
    
        if os.path.isdir(Main_Data_folder)==False:
            os.mkdir(Main_Data_folder)
           
        if os.path.isdir(Data_folder)==False:
            os.mkdir(Data_folder)
            
        Training_dir=os.path.join(Data_folder,"train")
        if os.path.isdir(Training_dir)==False:
            os.mkdir(Training_dir)
        Image_train_dir=os.path.join(Training_dir,"Image/class_0")
        if os.path.isdir( Image_train_dir)==False:
            os.makedirs( Image_train_dir)
        Label_train_dir=os.path.join(Training_dir,"Label/class_0")
        
        if os.path.isdir(Label_train_dir)==False:
            os.makedirs(Label_train_dir)
        
        Testing_dir=os.path.join(Data_folder,"test")
        if os.path.isdir(Testing_dir)==False:
            os.mkdir(Testing_dir)
            
        Image_test_dir=os.path.join(Testing_dir,"Image/class_0")
        if os.path.isdir( Image_test_dir)==False:
            os.makedirs( Image_test_dir)
            
        Label_test_dir=os.path.join(Testing_dir,"Label/class_0")
        if os.path.isdir( Label_test_dir)==False:
            os.makedirs( Label_test_dir)
    
        Validation_dir=os.path.join(Data_folder,"val")
        if os.path.isdir(Validation_dir)==False:
            os.mkdir(Validation_dir)
            
        Image_val_dir=os.path.join(Validation_dir,"Image/class_0")
        if os.path.isdir( Image_val_dir)==False:
            os.makedirs( Image_val_dir)
        
        Label_val_dir=os.path.join(Validation_dir,"Label/class_0")
        if os.path.isdir( Label_val_dir)==False:
            os.makedirs( Label_val_dir)



if __name__ == "__main__":
    # Path_to_Annotation="/home/llong/ULS23/ULS23_Radboudumc_Bone/ULS23_Radboudumc_Bone/labels"
    # Path_to_CT="/home/llong/ULS23/ULS23_Radboudumc_Bone/ULS23_Radboudumc_Bone/images"

    # Path_to_Annotation = "/home/llong/CTPelvis1K/DATASET/Labels/"
    # Path_to_CT = "/home/llong/CTPelvis1K/DATASET/Images/"


    # Path_to_CT = "data/pancreas/Images"
    # Path_to_Annotation= "data/pancreas/Labels"

    Path_to_CT = "/home/llong/Rib-Frac/train-images/"
    Path_to_Annotation = "/home/llong/Rib-Frac/train-labels/"


    saving_folder = "/home/u887755/thesis/data/Rib_temp"

    generate_dataset(Path_to_CT, Path_to_Annotation, saving_folder, use_label_contrain=True)

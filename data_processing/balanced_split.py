import os
import shutil
from tqdm import tqdm
import numpy as np

folder = "data/Rib"

def listdirfull(path):
    return [os.path.join(path, file) for file in os.listdir(path)]

all_files = os.listdir(f'{folder}/Preprocessed_data/train/Image/class_0')

# split patient wise
patients = [file.split("_")[0] for file in all_files]

# split patient wise

counts = np.unique(patients, return_counts=True)

total_scans = np.sum(counts[1])


accepted_split = False


while not accepted_split:



    # random sample 80% of patients
    train_patients = np.random.choice(counts[0], int(0.8*len(counts[0])), replace=False)
    remaining_patients = np.setdiff1d(counts[0], train_patients)
    # split remaining 20% into validation and test
    validation_patients = np.random.choice(remaining_patients, int(0.5*len(remaining_patients)), replace=False)
    test_patients = np.setdiff1d(remaining_patients, validation_patients)


    # print number of train files
    train_files = [file for file in all_files if file.split("_")[0] in train_patients]
    validation_files = [file for file in all_files if file.split("_")[0] in validation_patients]
    test_files = [file for file in all_files if file.split("_")[0] in test_patients]


    print(f"Total scans: {total_scans}")
    print(f"Train scans: {len(train_files)} ({len(train_files)/total_scans*100:.2f}%)")
    print(f"Validation scans: {len(validation_files)} ({len(validation_files)/total_scans*100:.2f}%)")   
    print(f"Test scans: {len(test_files)} ({len(test_files)/total_scans*100:.2f}%)")

    if len(train_files)/total_scans > 0.799 and len(train_files)/total_scans < 0.801 and len(validation_files)/total_scans > 0.099 and len(validation_files)/total_scans < 0.101 and len(test_files)/total_scans > 0.099 and len(test_files)/total_scans < 0.101:
        accepted_split = True


# move files to correct folders
for file in tqdm(validation_files):
    shutil.move(f'{folder}/Preprocessed_data/train/Image/class_0/{file}', f'{folder}/Preprocessed_data/val/Image/class_0/{file}')
    shutil.move(f'{folder}/Preprocessed_data/train/Label/class_0/{file}', f'{folder}/Preprocessed_data/val/Label/class_0/{file}')

for file in tqdm(test_files):
    shutil.move(f'{folder}/Preprocessed_data/train/Image/class_0/{file}', f'{folder}/Preprocessed_data/test/Image/class_0/{file}')
    shutil.move(f'{folder}/Preprocessed_data/train/Label/class_0/{file}', f'{folder}/Preprocessed_data/test/Label/class_0/{file}')

    
# print(all_files)
    


# files = os.listdir('data/Pelvis_full/Preprocessed_data/train/Image/class_0')
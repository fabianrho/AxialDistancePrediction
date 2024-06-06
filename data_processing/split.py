# move 10 percent of the data in data/Pelvis_full/Preprocessed_data/train to data/Pelvis_full/Preprocessed_data/val

import os
import shutil
from tqdm import tqdm

files = os.listdir('data/Pelvis_full/Preprocessed_data/train/Image/class_0')
files.sort()
subset = files[:int(len(files)*0.1)]
# remove the files from the files list
for file in subset:
    files.remove(file)

patient_ids_train = [file.split('_')[0] for file in files]
patient_ids_train = list(set(patient_ids_train))

patient_ids_subset = [file.split('_')[0] for file in subset]
patient_ids_subset = list(set(patient_ids_subset))
intersection = set(patient_ids_train) & set(patient_ids_subset)

files_to_move = [file for file in files if file.split('_')[0] in intersection]

for file in files_to_move:
    subset.append(file)

patient_ids_train = [file.split('_')[0] for file in files]
patient_ids_train = list(set(patient_ids_train))

patient_ids_subset = [file.split('_')[0] for file in subset]
patient_ids_subset = list(set(patient_ids_subset))
intersection = set(patient_ids_train) & set(patient_ids_subset)

intersection = set(patient_ids_train) & set(patient_ids_subset)

print((intersection))

# move files with id in intersection from train to val




# for file in tqdm(subset):
#     shutil.move(f'/home/u887755/thesis/data/PET_CT/Preprocessed_data/train/Image/class_0/{file}', f'/home/u887755/thesis/data/PET_CT/Preprocessed_data/val/Image/class_0/{file}')
#     shutil.move(f'/home/u887755/thesis/data/PET_CT/Preprocessed_data/train/Label/class_0/{file}', f'/home/u887755/thesis/data/PET_CT/Preprocessed_data/val/Label/class_0/{file}')
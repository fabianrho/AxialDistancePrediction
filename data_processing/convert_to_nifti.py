import dicom2nifti
import os
from tqdm import tqdm



dicom_dir = "data/pancreas_dicom/"
dicom_files = sorted(os.listdir(dicom_dir))

for i, file in enumerate(dicom_files):
    if not os.path.isdir(dicom_dir + file):
        continue
    patient = file.split("_")[1]
    output_file = "data/pancreas/" + patient + ".nii.gz"


    dicom2nifti.dicom_series_to_nifti(os.path.join(dicom_dir, file, os.listdir(os.path.join(dicom_dir, file))[0]), output_file)
    


# output_dir = "data/pancreas/0001.nii.gz"

# dicom2nifti.dicom_series_to_nifti(dicom_dir, output_dir)
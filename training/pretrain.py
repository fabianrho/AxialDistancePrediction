"""
General script for pretraining Swin-Transformer using self-supervised learning methods
"""

import sys
sys.path.append(".")
import torch
from models.swin_unet_v2 import SwinTransformerV2
from SSL.SSL_pretraining import SSL_pretraining
import argparse


def int_or_string(value):
    try:
        return int(value)
    except ValueError:
        if value == "all":
            return value
        else:
            argparse.ArgumentTypeError(f"{value} is not a valid integer or 'all'")

def none_or_int(value):
    if value == "None":
        return None
    else:
        return int(value)

parser = argparse.ArgumentParser(description="Pretraining Swin-Transformer using self-supervised learning")

parser.add_argument("--model_type", type=str, default="swin_unetv2", help="Model type")
parser.add_argument("--SSL_method", type=str, default="distance", help="Self-supervised learning method")
parser.add_argument("--dataset", type=str, default="Rib", help="Dataset")
parser.add_argument("--savefolder_id", type=str, default="", help="ID for save folder if needed")
parser.add_argument("--pairs_per_patient", type=int_or_string, default=100, help="Number of pairs per patient used for axial distance SSL method")
parser.add_argument("--max_patients", type=none_or_int, default="None", help="Maximum number of patients to use for pretraining")
parser.add_argument("--max_distance_cm", type=none_or_int, default="None", help="Maximum distance in cm between slices for axial distance SSL method")
parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")






args = parser.parse_args()



model_type = args.model_type
SSL_method = args.SSL_method
dataset = args.dataset
savefolder_id = args.savefolder_id
pairs_per_patient = args.pairs_per_patient
max_patients = args.max_patients
max_distance_cm = args.max_distance_cm
epochs = args.epochs
batch_size = args.batch_size


assert model_type == "swin_unetv2", "Only Swin-Transformer UNet V2 is supported for now"



match dataset:
    case "Rib":
        datafolder = "data/Rib/Preprocessed_data/train/Image/class_0"
        validation_datafolder = "data/Rib/Preprocessed_data/val/Image/class_0"
        nifti_dir = '/home/llong/Rib-Frac/train-images'


    case "Pelvis":
        datafolder = './data/Pelvis_full/Preprocessed_data/train/Image/class_0'
        validation_datafolder = './data/Pelvis_full/Preprocessed_data/val/Image/class_0'
        nifti_dir = '/home/llong/CTPelvis1K/DATASET/Images/'

    case _:
        raise ValueError("Dataset not supported")
    



backbone = SwinTransformerV2(img_size = 256, num_classes=1, drop_rate=0.2, window_size=8, depths = [2,2,2,2])

backbone.forward = backbone.encode


pretraining = SSL_pretraining(datafolder, validation_datafolder, nifti_dir, pairs_per_patient=pairs_per_patient, max_patients=max_patients, max_distance_cm = max_distance_cm, SSL_method=SSL_method, model_type=model_type, savefolder= f"pretrained_models/{model_type}/{dataset}/{SSL_method}", savefolder_id=savefolder_id)

pretraining.train(backbone, batch_size=batch_size, epochs = epochs, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))



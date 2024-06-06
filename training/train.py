"""
General training script for training models.
"""

import sys
sys.path.append(".")

from training.Training import Training
import torch


import argparse




parser = argparse.ArgumentParser(description="Training Swin-Unet")

parser.add_argument("--datafolder", type=str, default="../internship/256/data_256/Lung/axial/", help="Data folder")
parser.add_argument("--model_type", type=str, default="swin_unetv2", help="Model type")
parser.add_argument("--savefolder", type=str, default="trained_models/swin_unetv2/Lung", help="Save folder")
parser.add_argument("--savefolder_id", type=str, default="", help="ID for save folder if needed")

parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=24, help="Batch size")
parser.add_argument("--loss", type=str, default="bce", help="Loss function")
parser.add_argument("--flip_and_rotate_aug", action = "store_true", help="Whether to use random flip and rotate")

parser.add_argument("--pretrained_path", type=str, default=None, help="Path to pretrained model")
parser.add_argument("--mirror_enc_dec", action = "store_true", help="Pretrained encoder are mirrored to decoder")



args = parser.parse_args()


datafolder = args.datafolder
model_type = args.model_type
savefolder = args.savefolder
savefolder_id = args.savefolder_id
epochs = args.epochs
batch_size = args.batch_size
loss = args.loss
flip_and_rotate_aug = args.flip_and_rotate_aug
pretrained_path = args.pretrained_path
mirror_encoder_decoder = args.mirror_enc_dec




training = Training(datafolder=datafolder, 
                    batch_size=batch_size,
                    model_type=model_type,
                    savefolder = savefolder,
                    savefolder_id=savefolder_id,
                    )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
training.train(device=device, epochs = epochs, loss = loss, pretrained_path=pretrained_path, mirror_encoder_decoder=mirror_encoder_decoder, flip_and_rotate_aug=flip_and_rotate_aug)

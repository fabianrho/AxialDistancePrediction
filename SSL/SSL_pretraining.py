import sys
sys.path.append(".")
sys.path.append("./SSL")

import os

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np

from tqdm import tqdm
import copy

from models.swin_unet_v1 import SwinTransformerSys
from models.swin_unet_v2 import SwinTransformerV2

from SSL.axial_distance_loader import AxialDistanceDataset


from SSL.SSL_methods import BYOL, BarlowTwins, DINO, SimCLR, MoCo, AxialDistanceProjection

from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.transforms.moco_transform import MoCoV2Transform
from lightly.transforms.byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)

from lightly.loss import NTXentLoss, NegativeCosineSimilarity, BarlowTwinsLoss,DINOLoss, DCLLoss
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule

from lightly.transforms.dino_transform import DINOTransform


from SSL.get_dataloader import get_dataloader

from SSL.simmim import *
from SSL.simmim_dataloader import *


class SSL_pretraining():

    def __init__(self, datafolder, validation_datafolder = None, nifti_dir = None, pairs_per_patient = 100, max_patients = None, max_distance_cm = None, SSL_method = "byol", model_type = "swin_unet", savefolder="pretrained_models", savefolder_id = ""):
        self.datafolder = datafolder
        self.validation_datafolder = validation_datafolder
        self.nifti_dir = nifti_dir
        self.pairs_per_patient = pairs_per_patient # only used for axial distance prediction
        self.max_patients = max_patients # only used for axial distance prediction
        self.max_distance_cm = max_distance_cm # only used for axial distance prediction

        self.savefolder = savefolder

        self.SSL_method = SSL_method
        self.model_type = model_type

        self.savefolder = os.path.join(self.savefolder, SSL_method + "_" + model_type + "_" + savefolder_id)

        if not os.path.exists(self.savefolder):
            os.makedirs(self.savefolder)
        else:
            # check if folder is empty
            if len(os.listdir(self.savefolder)) > 0 and os.path.exists(os.path.join(self.savefolder, "settings.txt")) and "temp" not in savefolder_id:
                raise ValueError("Save folder already exists")
            if os.path.exists(self.savefolder) and "temp" in savefolder_id:
                Warning("Overwriting temp folder")

        self.checkpoint_folder = os.path.join(self.savefolder, "checkpoints")
        if not os.path.exists(self.checkpoint_folder):
            os.mkdir(self.checkpoint_folder)

        




    def load_data(self, transform, batch_size):

        match self.SSL_method:
            case "distance":

                assert self.nifti_dir is not None, "nifti_dir must be provided for axial distance prediction" 
                assert self.validation_datafolder is not None, "validation_datafolder must be provided for axial distance prediction"
                train_data = AxialDistanceDataset(self.datafolder, 
                                        self.nifti_dir, 
                                        max_patients = self.max_patients,
                                        pairs_per_patient = self.pairs_per_patient, 
                                        max_distance_cm = self.max_distance_cm,
                                        resize_to=(256,256), 
                                        create_three_channels=True)

                val_data = AxialDistanceDataset(self.validation_datafolder,
                                        self.nifti_dir,
                                        validation = True,
                                        max_patients = self.max_patients,
                                        # pairs_per_patient = self.pairs_per_patient, 
                                        pairs_per_patient = 200, # fix to 200 pairs per patient for validation, fair comparison between 100 and 200 training pairs per patient

                                        max_distance_cm = self.max_distance_cm,
                                        resize_to=(256,256), 
                                        create_three_channels=True)

                self.dataloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
                val_data.max_height = train_data.max_height
                self.max_height = train_data.max_height
                self.val_dataloader = data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)

            case "simmim":
                if "/class_0" in self.datafolder:
                    self.datafolder = self.datafolder.replace("/class_0", "")
                self.dataloader = build_loader_simmim(path=self.datafolder, batch_size=batch_size, num_workers=4, mask_ratio=0.6)



            case _:
                self.dataloader = get_dataloader(self.datafolder, transform=transform, batch_size=batch_size)


    def train(self, backbone, batch_size = 224, epochs = 250, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

        self.device = device

        if isinstance(backbone, SwinTransformerSys):
            backbone = self.swin_unet_init_weights(backbone)

        elif isinstance(backbone, SwinTransformerV2):
            if self.SSL_method == "simmim":
                self.model = build_simmim()
            else:
                backbone = self.swin_unet_init_weights(backbone, pretrained_path="./pretrained/swinv2_tiny_patch4_window8_256.pth")


        with open(os.path.join(self.savefolder, "settings.txt"), "w") as f:
            attributes = vars(self)
            for attribute_name, attribute_value in attributes.items():
                f.writelines(f"{attribute_name}: {attribute_value} \n")
            f.writelines(f"batch_size: {batch_size} \n")
            f.writelines(f"epochs: {epochs} \n")


        match self.SSL_method:
            # generative
            case "simmim":
                self.train_simmim(batch_size, epochs)
            # contrastive
            case "byol":
                self.train_byol(backbone, batch_size, epochs)
            case "simclr":
                self.train_simclr(backbone, batch_size, epochs)
            case "dcl":
                self.train_simclr(backbone, batch_size, epochs)

            case "distance":
                self.train_distance_prediction(backbone, batch_size, epochs)


            case _:
                raise ValueError("Invalid SSL method")


    def train_distance_prediction(self, backbone, batch_size, epochs):
        self.load_data(transform=None, batch_size=batch_size)
        self.model = AxialDistanceProjection(backbone)

        optimizer = torch.optim.Adam([
            {'params': self.model.backbone.parameters(), 'lr': 0.001},  # Regular learning rate for encoder
            {'params': self.model.projection_head.parameters(), 'lr': 0.01 * 0.0001}  # Lower LR for pred. head
            ])
        
        self.model.to(self.device)
        criterion = torch.nn.MSELoss()

        for self.epoch in range(1, epochs+1):
            self.train_losses = []
            with tqdm(self.dataloader, unit="batch", ascii=" >=") as tepoch:
                self.model.train()
                for batch in tepoch:
                    optimizer.zero_grad()
                    data1, data2, distance_mm = batch
                    data1 = data1.to(self.device)
                    data2 = data2.to(self.device)
                    distance_mm = distance_mm.to(self.device)



                    output = self.model(data1, data2)


                    loss = criterion(output, distance_mm.float().unsqueeze(1))

                    loss.backward()
                    optimizer.step()
                    self.train_losses.append(loss.item())
                    tepoch.set_postfix(loss=loss.item())#, avg_loss=np.mean(epoch_losses))
            self.train_loss = torch.mean(torch.tensor(self.train_losses))# / len(self.dataloader)
                

            self.model.eval()
            with torch.no_grad():
                self.val_losses = []
                for batch in self.val_dataloader:
                    data1, data2, distance_mm = batch
                    data1 = data1.to(self.device)
                    data2 = data2.to(self.device)
                    distance_mm = distance_mm.to(self.device)

                    output = self.model(data1, data2)

                    loss = criterion(output, distance_mm.float().unsqueeze(1))
                    self.val_losses.append(loss.item())

            self.val_loss = torch.mean(torch.tensor(self.val_losses))
            print(f"epoch: {self.epoch:>02}, train loss: {self.train_loss:.5f}, val loss: {self.val_loss:.5f}")

            self.on_epoch_end(use_val_loss=True)

        torch.save(self.model.backbone.state_dict(), os.path.join(self.savefolder, f"{self.SSL_method}.pth"))
        



    def train_simmim(self, batch_size, epochs):

        self.load_data(transform=None, batch_size=batch_size)
        self.model.to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), eps = 1e-8, betas = (0.9, 0.999), lr = 5e-4, weight_decay = 0.05)

        self.model.train()

        for self.epoch in range(1, epochs+1):
            self.train_losses = []
            with tqdm(self.dataloader, unit = "batch", ascii=" >=", ) as tepoch:
                for img, mask,_ in tepoch:
                    optimizer.zero_grad()

                    img = img.to(self.device)
                    mask = mask.to(self.device)
                    loss = self.model(img, mask)
                    self.train_losses.append(loss)

                    loss.backward()

                    optimizer.step()
                    tepoch.set_postfix(loss = loss.item())
                self.train_loss = torch.mean(torch.tensor(self.train_losses))# / len(self.dataloader)
                print(f"epoch: {self.epoch:>02}, loss: {self.train_loss:.5f}")

                self.on_epoch_end()

        torch.save(self.model.backbone.state_dict(), os.path.join(self.savefolder, f"{self.SSL_method}.pth"))



    def train_byol(self, backbone, batch_size, epochs):
        transform = BYOLTransform(
        view_1_transform=BYOLView1Transform(input_size=256),
        view_2_transform=BYOLView2Transform(input_size=256),
        )

        self.load_data(transform, batch_size)

        criterion = NegativeCosineSimilarity()
        optimizer = torch.optim.SGD(backbone.parameters(), lr=0.06)

        self.model = BYOL(backbone)
        self.model.to(self.device)


        for self.epoch in range(1,epochs + 1):
            self.train_losses = []

            momentum_val = cosine_schedule(self.epoch, epochs, 0.996, 1)
            with tqdm(self.dataloader, unit = "batch", ascii=" >=", ) as tepoch:

                for batch in tepoch:
                    x0, x1 = batch[0]
                    update_momentum(self.model.backbone, self.model.backbone_momentum, m=momentum_val)
                    update_momentum(
                        self.model.projection_head, self.model.projection_head_momentum, m=momentum_val
                    )
                    x0 = x0.to(self.device)
                    x1 = x1.to(self.device)
                    p0 = self.model(x0)
                    z0 = self.model.forward_momentum(x0)
                    p1 = self.model(x1)
                    z1 = self.model.forward_momentum(x1)
                    loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))
                    self.train_losses.append(loss)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    tepoch.set_postfix(loss=loss.item())
                self.train_loss = torch.mean(torch.tensor(self.train_losses))# / len(self.dataloader)

                print(f"epoch: {self.epoch:>02}, loss: {self.train_loss:.5f}")
                self.on_epoch_end()


        torch.save(self.model.backbone.state_dict(), os.path.join(self.savefolder, f"{self.SSL_method}.pth"))


    def train_simclr(self, backbone, batch_size, epochs):

        transform = SimCLRTransform(input_size = 256)
        self.load_data(transform, batch_size)

        self.model = SimCLR(backbone)
        self.model.to(self.device)

        match self.SSL_method:
            case "simclr":  
                criterion = NTXentLoss()
            case "dcl":
                criterion = DCLLoss()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.06)

        for self.epoch in range(1, epochs+1):
            self.train_losses = []
            # total_loss = 0
            with tqdm(self.dataloader, unit = "batch", ascii=" >=", ) as tepoch:
                for batch in tepoch:
                    x0, x1 = batch[0]
                    x0 = x0.to(self.device)
                    x1 = x1.to(self.device)
                    z0 = self.model(x0)
                    z1 = self.model(x1)
                    loss = criterion(z0, z1)
                    self.train_losses.append(loss)
                    # total_loss += loss.detach()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    tepoch.set_postfix(loss=loss.item())
                    

                self.train_loss = torch.mean(torch.tensor(self.train_losses))# / len(self.dataloader)
                print(f"epoch: {self.epoch:>02}, loss: {self.train_loss:.5f}")

                self.on_epoch_end()
        torch.save(self.model.backbone.state_dict(), os.path.join(self.savefolder, f"{self.SSL_method}.pth"))



    def on_epoch_end(self, use_val_loss = False):


        self.save_history(use_val_loss)
        self.save_checkpoint(use_val_loss)


    def save_history(self, use_val_loss = False):
        if os.path.exists(os.path.join(self.savefolder, "training_log.csv")):
            with open(os.path.join(self.savefolder, "training_log.csv"), "a") as f:
                if use_val_loss:
                    f.write(f"{self.epoch},{self.train_loss},{self.val_loss}\n")
                else:
                    f.write(f"{self.epoch},{self.train_loss}\n")
        else:
            with open(os.path.join(self.savefolder, "training_log.csv"), "w") as f:
                if use_val_loss:
                    f.write("Epoch,Train_loss,Val_loss\n")
                    f.write(f"{self.epoch},{self.train_loss},{self.val_loss}\n")
                else:
                    f.write("Epoch,Loss\n")
                    f.write(f"{self.epoch},{self.train_loss}\n")


    def save_checkpoint(self, use_val_loss):
        """saves the model to a checkpoint file
        """
        if use_val_loss:
            if self.epoch == 1:  
                self.best_loss = self.val_loss
            else:
                if self.val_loss < self.best_loss:
                    self.best_loss = self.val_loss
                    torch.save(self.model.backbone.state_dict(), os.path.join(self.checkpoint_folder, f"epoch_{self.epoch}_{self.best_loss:.3f}.pth"))
                    torch.save(self.model.state_dict(), os.path.join(self.checkpoint_folder, f"best_full.pth"))
        else:
            if self.epoch == 1:  
                self.best_loss = self.train_loss
            else:
                if self.train_loss < self.best_loss:
                    self.best_loss = self.train_loss
                    torch.save(self.model.backbone.state_dict(), os.path.join(self.checkpoint_folder, f"epoch_{self.epoch}_{self.best_loss:.3f}.pth"))
                    torch.save(self.model.state_dict(), os.path.join(self.checkpoint_folder, f"best_full.pth"))





    def swin_unet_init_weights(self, backbone, pretrained_path = "/home/u887755/thesis/pretrained/swin_tiny_patch4_window7_224.pth"):
        # pretrained_path = "/home/u887755/thesis/pretrained/swin_tiny_patch4_window7_224.pth"
        # pretrained_path = "swin_byol_im.pth"

        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            pretrained_dict = torch.load(pretrained_path, map_location=self.device)

            if "model" not in pretrained_dict:
                print("---start load pretrained model by splitting---")
                # print("pretrained_dict.keys():{}".format(pretrained_dict.keys()))
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]

                
                msg = backbone.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained model of swin encoder---")

            model_dict = backbone.state_dict()

            full_dict = copy.deepcopy(pretrained_dict)

            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]


            msg = backbone.load_state_dict(full_dict, strict=False)

        return backbone
    




if __name__ == "__main__":
    # datafolder = "../internship/256/data_256/Lung/axial/"
    datafolder = "data/Pelvis/Preprocessed_data/train/Image/class_0"
    val_datafolder = "data/Pelvis/Preprocessed_data/val/Image/class_0"

    savefolder_id = "pelvis"

    model_type = "swin_unetv2"
    backbone = SwinTransformerSys(num_classes = 1)
    backbone.forward = backbone.encode


    pretraining = SSL_pretraining(datafolder, SSL_method="distance", model_type=model_type, savefolder_id=savefolder_id)

    pretraining.train(backbone, batch_size=128, device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu"))






    





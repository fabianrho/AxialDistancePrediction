"""
This file contains the training class for training segmentation models with swin-unet models
"""

import sys
sys.path.append(".")


import torch
import torch.nn as nn

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import copy


from models.UNet import UNet
from models.unetr import UNETR
from models.swin_unet_v1 import SwinTransformerSys
from models.swin_unet_v2 import SwinTransformerV2
from models.transbts import TransBTS

from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import swin_t, Swin_T_Weights

from utils.data_loading import DataLoaderSegmentation
from utils.metrics import Dice, DiceLoss





class Training():
    """Class for training segmentation models with vision transformers
    """

    def __init__(self, datafolder, savefolder="trained_models", model_type = "swin_unetv2", savefolder_id = "", batch_size = 24):
        """Initializes the training class

        Args:
            datafolder (str): folder containing the preprocessed data
            savefolder (str, optional): path to save training results folder to. Defaults to "trained_models".
            model_type (str, optional): segmentation model type to train. Defaults to "swin_unetv2".
            savefolder_id (str, optional): optional identifier to append to savefolder. Defaults to "".
            batch_size (int, optional): batch size. Defaults to 24.
        """

        # setup data paths
        self.datafolder = os.path.join(datafolder, "Preprocessed_data")
        self.image_data = os.path.join(self.datafolder, "train")
        self.label_data = os.path.join(self.datafolder, "train")

        self.image_data_val = os.path.join(self.datafolder, "val")
        self.label_data_val = os.path.join(self.datafolder, "val")

        

        self.model_type = model_type.lower() 
        assert self.model_type in ["swin_unet", "swin_unetv2"], "Model type not supported"


        self.savefolder = os.path.join(savefolder, model_type + "_" + savefolder_id)
        self.batch_size = batch_size


        # create savefolder if it does not exist, else raise error
        if not os.path.exists(self.savefolder):
            os.makedirs(self.savefolder)
        else:
            # check if folder is empty
            if len(os.listdir(self.savefolder)) > 0 and os.path.exists(os.path.join(self.savefolder, "settings.txt")) and "temp" not in savefolder_id and "temp" not in self.savefolder:
                raise ValueError("Save folder already exists")
            if os.path.exists(self.savefolder) and ("temp" in savefolder_id or "temp" in self.savefolder):
                # overwrite savefolder if temp in name, used for debugging purposes
                Warning("Overwriting temp folder")
            
        # create subfolders for checkpoints and validation visualisation
        self.checkpoint_folder = os.path.join(self.savefolder, "checkpoints")
        self.validation_visualisation_folder = os.path.join(self.savefolder, "validation_visualisation")
        if not os.path.exists(self.checkpoint_folder) and not os.path.exists(self.validation_visualisation_folder):
            os.mkdir(self.checkpoint_folder)
            os.mkdir(self.validation_visualisation_folder)
        



    def train(self, epochs = 500, n_augmentations = 1, loss = "bce", pretrained_path = None, mirror_encoder_decoder = True, flip_and_rotate_aug = False, use_crop = False, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        """ General training function for the segmentation model

        Args:
            epochs (int, optional): number of epochs to train for. Defaults to 500.
            n_augmentations (int, optional): Number of augmented samples to create per training image. Defaults to 1.
            loss (str, optional): loss function to use during training. Defaults to "bce".
            pretrained_path (str, optional): path to pretrained weights. Defaults to None.
            mirror_encoder_decoder (bool, optional): whether to copy the weights from the pretrained encoder to the decoder (inverted). Defaults to True.
            flip_and_rotate_aug (bool, optional): whether to use random flip and rotation augmentation rather than random affine transformation. Defaults to False.
            use_crop (bool, optional): whether to crop to specified image size, rather than resizing. Defaults to False.
            device (torch.device, optional): device to train with. Defaults to torch.device("cuda:0" if torch.cuda.is_available() else "cpu").
        """

        
        self.device = device # training device
        self.load_data(n_augmentations, flip_and_rotate_aug, use_crop) # create training and validation dataloaders
        use_imagenet = pretrained_path is None # use imagenet weights if no checkpoint is provided
        self.create_model(use_imagenet) # create the segmentation model

        # load pretrained weights if provided
        if pretrained_path is not None:

            # code used from https://github.com/HuCaoFighting/Swin-Unet
            if self.model_type == "swin_unetv2" and mirror_encoder_decoder:
                state_dict = torch.load(pretrained_path, map_location="cpu")
                new_dict = copy.deepcopy(state_dict)
                for k, v in state_dict.items():
                    if "layers." in k:
                        current_layer_num = 3-int(k.split(".")[1])
                        current_k = "layers_up." + str(current_layer_num) + k[8:]
                        if current_k in new_dict:
                            new_dict.update({current_k:v})

                # delete prediction head used for pretraining with SimMIM
                if "simmim" in pretrained_path:
                    del new_dict["output.weight"]

                self.model.load_state_dict(new_dict, strict=False)
                print("Copied pretrained weights from encoder to decoder")
            else:
                self.model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))

        self.model.to(self.device)

        # print(f"Model location: {self.device}")


        # optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)

        # loss function
        if loss == "dice":
            self.criterion = DiceLoss(average='micro', num_classes=1)
        elif loss == "bce":
            self.criterion = nn.BCELoss()

        # dice score function
        self.dice = Dice(average='micro', num_classes=1)
        self.epochs = epochs


        # save settings to savefolder for reference
        with open(os.path.join(self.savefolder, "settings.txt"), "w") as f:
            attributes = vars(self)
            for attribute_name, attribute_value in attributes.items():
                f.writelines(f"{attribute_name}: {attribute_value} \n")

            f.writelines(f"loss: {repr(loss)} \n")
            f.writelines(f"optimizer: {repr(optimizer)} \n")
            f.writelines(f"pretrained_path: {pretrained_path} \n")
            f.writelines(f"flip_and_rotate_aug: {flip_and_rotate_aug} \n")



        
        # training loop
        for self.epoch in range(1, self.epochs+1):
            self.model.train() # enable training mode

            # lists to save training loss and dice score
            self.train_dices = [] 
            self.train_losses = []

            # epoch loop
            with tqdm(self.train_loader, unit = "batch", ascii=" >=", ) as tepoch:
                # loop over batches in the training data
                for batch in tepoch:
                    data, label = batch
                    data, label = data.to(self.device), label.to(self.device) # move data and label to device (GPU)

                    optimizer.zero_grad() # reset gradients

                    output = self.model(data) # prediction

                    # apply sigmoid if using swin transformer
                    if self.model_type == "swin_unet" or self.model_type == "swin_unetv2":
                        output = torch.sigmoid(output)
                    
                    # calculate loss and dice score on training batch
                    loss = self.criterion(output, label)
                    self.train_losses.append(loss)
                    dice_score = self.dice(output, label)
                    self.train_dices.append(dice_score)
                    
                    # update weights
                    loss.backward()
                    optimizer.step()
                    
                    # update progress bar
                    tepoch.set_postfix(loss=torch.mean(torch.tensor(self.train_losses)).item(), dice=torch.mean(torch.tensor(self.train_dices)).item(), dice_moving_average = torch.mean(torch.tensor(self.train_dices[-10:])).item())

            # callback functions for validation data
            self.on_epoch_end()

    
            
    def on_epoch_end(self):
        """ Function that is called at the end of each epoch
        """
        self.validation()
        self.save_checkpoint()
        self.save_history()
        if self.epoch > 1:
            self.plot_history()
        self.plot_validation()

    def validation(self):
        """ Callback function to calculate the validation loss and dice score after each epoch
        """
        self.model.eval()
        with torch.no_grad():
            self.val_dices = []
            self.val_losses = []
            for batch in self.val_loader:
                data, label = batch
                data, label = data.to(self.device), label.to(self.device)
                output = self.model(data)
                if self.model_type == "swin_unet" or self.model_type == "swin_unetv2":
                    output = torch.sigmoid(output)
                loss = self.criterion(output, label)
                dice_score = self.dice(output, label)
                self.val_dices.append(dice_score)
                self.val_losses.append(loss)
            print(f"Epoch: {self.epoch}/{self.epochs} | Train Loss: {torch.mean(torch.tensor(self.train_losses)):.4f}, Train Dice: {torch.mean(torch.tensor(self.train_dices)):.4f} | Validation Loss: {torch.mean(torch.tensor(self.val_losses)):.4f}, Validation Dice: {torch.mean(torch.tensor(self.val_dices)):.4f} \n")




    def save_checkpoint(self):
        """ Callback function to save the model weights if the loss or dice improves on the validation set
        """
        if self.epoch == 1:  
            self.best_loss = torch.mean(torch.tensor(self.val_losses))
            self.best_dice = torch.mean(torch.tensor(self.val_dices))
        else:
            if torch.mean(torch.tensor(self.val_losses)) < self.best_loss:
                self.best_loss = torch.mean(torch.tensor(self.val_losses))
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_folder, f"epoch_{self.epoch}_best_loss_{self.best_loss:.3f}.pth"))
            if torch.mean(torch.tensor(self.val_dices)) > self.best_dice:
                self.best_dice = torch.mean(torch.tensor(self.val_dices))
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_folder, f"epoch_{self.epoch}_best_dice_{self.best_dice:.3f}.pth"))

    def save_history(self):
        """ Callback function to save the training history to a csv file
        """
        if os.path.exists(os.path.join(self.savefolder, "training_log.csv")):
            with open(os.path.join(self.savefolder, "training_log.csv"), "a") as f:
                f.write(f"{self.epoch},{torch.mean(torch.tensor(self.train_losses))},{torch.mean(torch.tensor(self.train_dices))},{torch.mean(torch.tensor(self.val_losses))},{torch.mean(torch.tensor(self.val_dices))}\n")
        else:
            with open(os.path.join(self.savefolder, "training_log.csv"), "w") as f:
                f.write("Epoch,Train Loss,Train Dice,Validation Loss,Validation Dice\n")
                f.write(f"{self.epoch},{torch.mean(torch.tensor(self.train_losses))},{torch.mean(torch.tensor(self.train_dices))},{torch.mean(torch.tensor(self.val_losses))},{torch.mean(torch.tensor(self.val_dices))}\n")


    def plot_history(self):
        """ Callback function to plot the training history
        """

        history = np.genfromtxt(os.path.join(self.savefolder, "training_log.csv"), delimiter=",", skip_header=1)
        fig, ax = plt.subplots(2, 1, figsize=(15, 15))
        # increase font size
        plt.rcParams.update({'font.size': 18})
        # make x axis integer
        ax[0].plot(history[:, 0], history[:, 1], label="Train Loss")
        ax[0].plot(history[:, 0], history[:, 3], label="Validation Loss")
        ax[0].set_title("Loss")
        ax[0].legend()
        ax[0].xaxis.get_major_locator().set_params(integer=True)

        ax[1].plot(history[:, 0], history[:, 2], label="Train Dice")
        ax[1].plot(history[:, 0], history[:, 4], label="Validation Dice")
        ax[1].set_title("Dice")
        ax[1].legend()
        ax[1].xaxis.get_major_locator().set_params(integer=True)

        plt.savefig(os.path.join(self.savefolder, "history.png"))
        plt.close(fig)



    def plot_validation(self):
        """ Callback function to plot the example predictions on five samples from the validation data
        """

        # take the first batch, no shuffling so always retrieves the same batch
        for data, label in self.val_loader:
            break

        output = self.model(data.to(self.device))

        fig, ax = plt.subplots(5, 3, figsize=(15, 15))
        fig.suptitle('Epoch %s'%self.epoch, fontsize=18)

        for i in range(5):

        # plotting validation sample

            plotting_data = data[i].cpu().numpy()
            plotting_data = np.transpose(plotting_data, (1, 2, 0))

            # check if data is single channel
            if plotting_data.shape[-1] == 1:
                plotting_data = np.concatenate((plotting_data, plotting_data, plotting_data), axis=-1)

            plotting_ground_truth = label[i].cpu().numpy()
            plotting_ground_truth = np.transpose(plotting_ground_truth, (1, 2, 0))

            # check if data is single channel
            if plotting_ground_truth.shape[-1] == 1:
                plotting_ground_truth = np.concatenate((plotting_ground_truth, plotting_ground_truth, plotting_ground_truth), axis=-1)

            if self.model_type == "swin_unet" or self.model_type == "swin_unetv2":
                plotting_label = torch.sigmoid(output[i]).cpu().detach().numpy()
            else:
                plotting_label = output[i].cpu().detach().numpy()
            plotting_label = np.transpose(plotting_label, (1, 2, 0))
            plotting_label = np.concatenate((plotting_label, plotting_label, plotting_label), axis=-1)

            ax[i,0].imshow(plotting_data, cmap='gray')
            ax[i,1].imshow(plotting_ground_truth, cmap='gray')
            ax[i,2].imshow(plotting_label, cmap='gray')
            # no ticks
            ax[i,0].axis('off')
            ax[i,1].axis('off')
            ax[i,2].axis('off')
        plt.savefig(f"{self.validation_visualisation_folder}/{self.epoch}.png")

        plt.close(fig)


    def load_data(self, n_augmentations, use_crop, flip_and_rotate_aug):
        """Create dataloaders for training and validation data

        Args:
            n_augmentations (int): number of augmented samples to create per image
            use_crop (bool): whether to crop to specified size, rather than resizing
            flip_and_rotate_aug (bool): whether to use random flip and rotate augmentation rather than random affine augmentation
        """

        # specific settings for different models, given pretrained weights on imagenet
        if self.model_type in ["swin_unet"]:
            resize_to = 224
            create_three_channels = True
        elif self.model_type in ["swin_unetv2"]:
            resize_to = 256
            create_three_channels = True
        else:
            resize_to = None
            create_three_channels = False

        self.training_data = DataLoaderSegmentation(self.image_data, augmentation=True, flip_and_rotate_aug=flip_and_rotate_aug, n_augmentations=n_augmentations, create_three_channels=create_three_channels, resize_to=resize_to, use_crop=use_crop)
        self.train_loader = torch.utils.data.DataLoader(self.training_data, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

        self.validation_data = DataLoaderSegmentation(self.image_data_val, augmentation=False, create_three_channels=create_three_channels, resize_to=resize_to, use_crop=use_crop)
        self.val_loader = torch.utils.data.DataLoader(self.validation_data, batch_size=self.batch_size, shuffle=False, num_workers=0, drop_last=False)


    def create_model(self, use_imagenet = True):
        """ Initializes the specified model

        Args:
            use_imagenet (bool, optional): whether to initialize with pretrained imagenet weights. Defaults to True.
        """

        match self.model_type:

            case "unet":
                self.model = UNet(n_channels=1, n_classes=1, n_filters_base=16)

            case "swin_unet":
                self.model = SwinTransformerSys(num_classes=1, drop_rate=0.2)
                if use_imagenet:
                    self.swin_unet_init_weights()

            case "swin_unetv2":
                self.model = SwinTransformerV2(img_size = 256, num_classes=1, drop_rate=0.2, window_size=8, depths = [2,2,2,2])
                if use_imagenet:
                    self.swin_unet_init_weights(pretrained_path="./pretrained/swinv2_tiny_patch4_window8_256.pth")



    def swin_unet_init_weights(self, pretrained_path = "/home/u887755/thesis/pretrained/swin_tiny_patch4_window7_224_22k.pth"):
        """ Initializes the weights of the swin transformer model with the pretrained weights on imagenet
            Code used from https://github.com/microsoft/Swin-Transformer 

        Args:
            pretrained_path (str, optional): path to weights file. Defaults to "/home/u887755/thesis/pretrained/swin_tiny_patch4_window7_224_22k.pth".
        """

    
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

                
                msg = self.model.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained model of swin encoder---")

            model_dict = self.model.state_dict()

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


            msg = self.model.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")

        



if __name__ == "__main__":
   pass
    
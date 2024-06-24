import sys
sys.path.append(".")
sys.path.append("..")

import torch
import torch.nn as nn
from tqdm import tqdm
import os.path as path

import numpy as np
from skimage import measure
import glob
from monai.metrics import compute_hausdorff_distance


from models.swin_unet_v2 import SwinTransformerV2

from utils.data_loading import DataLoaderSegmentation
from utils.metrics import Dice

import pandas as pd

import matplotlib.pyplot as plt


def lesion_detection_from_predictions(predictions_path, threshold = 0.5, one_per_lesion = False, exclude_multilesion_samples = False):

    split = predictions_path.split("/")[-1].split("_")[1]
    data_path = f"/home/u887755/internship/256/data_256/Lung/axial/Preprocessed_data/{split}"
    data = DataLoaderSegmentation(data_path, create_three_channels=True)

    dice = Dice()

    predictions = torch.load(predictions_path, map_location="cpu")
    ground_truth = torch.load(f"{path.join(*predictions_path.split('/')[:-1])}/groundtruth_{split}.pt", map_location = "cpu")

    predictions[predictions > threshold] = 1
    predictions[predictions <= threshold] = 0

    data_files = data.img_files

    n_detected = 0
    n_non_detected = 0

    detected_lesions= {}
    lesion_sizes = []
    skipped = 0


    results = pd.DataFrame()

    for i in range(predictions.shape[0]):
        output = predictions[i].cpu().squeeze(0).numpy()
        mask = ground_truth[i].cpu().squeeze(0).numpy()

        if one_per_lesion:
            data_file = data_files[i].split("/")[-1].split("_")[1]
        else:
            data_file = data_files[i]
        


        output[output > threshold] = 1
        output[output <= threshold] = 0

        
        props_mask = measure.regionprops(label_image = measure.label(mask.astype(np.uint8)))


        if exclude_multilesion_samples and len(props_mask) > 1:
            skipped += 1
            continue


        for j, prop in enumerate(props_mask):

            if len(props_mask) > 1:
                data_file = f"{data_files[i]}_{j}"
            if data_file not in detected_lesions:
                detected_lesions[data_file] = 0
                if one_per_lesion:
                    lesion_sizes.append(props_mask[0].area)


            cropped_mask = np.expand_dims(mask, -1).astype(np.uint8)[prop.bbox[0]:prop.bbox[2], prop.bbox[1]:prop.bbox[3]]
            cropped_output = np.expand_dims(output, -1)[prop.bbox[0]:prop.bbox[2], prop.bbox[1]:prop.bbox[3]]

            detected = int(np.sum(cropped_mask *cropped_output) > 0)

            if not one_per_lesion:
                lesion_sizes.append(prop.area)

            if detected:
                n_detected += 1
                detected_lesions[data_file] = 1
            else:
                n_non_detected += 1

    results["file"] = detected_lesions.keys()
    results["detected"] = detected_lesions.values()
    results["area"] = lesion_sizes

    return results

def lesion_detection(model, split= "val", threshold = 0.5, one_per_lesion = False, gpu = "cuda:1"):
    data_path = f"/home/u887755/internship/256/data_256/Lung/axial/Preprocessed_data/{split}"
    data = DataLoaderSegmentation(data_path, create_three_channels=True)
    dataloader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=False)
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")

    model.eval()
    model = model.to(device)

    n_detected = 0
    n_non_detected = 0

    detected_lesions= {}
    lesion_sizes = []
    skipped = 0

    predictions = []
    ground_truth = []

    results = pd.DataFrame()

    # make predictions
    for batch in tqdm(dataloader):
        # break
        img, mask = batch
        img = img.to(device)
        mask = mask.to(device)


        with torch.no_grad():
            output = torch.sigmoid(model(img))

        predictions.append(output)
        ground_truth.append(mask)

    all_predictions = torch.cat(predictions, dim=0)
    all_ground_truth = torch.cat(ground_truth, dim=0)

    data_files = data.img_files

    for i in range(all_predictions.shape[0]):
        output = all_predictions[i].cpu().squeeze(0).numpy()
        mask = all_ground_truth[i].cpu().squeeze(0).numpy()

        if one_per_lesion:
            data_file = data_files[i].split("/")[-1].split("_")[1]
        else:
            data_file = data_files[i]
        


        output[output > threshold] = 1
        output[output <= threshold] = 0

        
        props_mask = measure.regionprops(label_image = measure.label(mask.astype(np.uint8)))

        if len(props_mask) > 1:
            skipped += 1
            continue

        if data_file not in detected_lesions:
            detected_lesions[data_file] = 0
            if one_per_lesion:
                lesion_sizes.append(props_mask[0].area)
        for prop in props_mask:
            cropped_mask = np.expand_dims(mask, -1).astype(np.uint8)[prop.bbox[0]:prop.bbox[2], prop.bbox[1]:prop.bbox[3]]
            cropped_output = np.expand_dims(output, -1)[prop.bbox[0]:prop.bbox[2], prop.bbox[1]:prop.bbox[3]]
            lesion_sizes.append(prop.area)

            detected = int(np.sum(cropped_mask *cropped_output) > 0)

            if not one_per_lesion:
                lesion_sizes.append(prop.area)

            if detected:
                n_detected += 1
                detected_lesions[data_file] = 1
            else:
                n_non_detected += 1

    results["file"] = detected_lesions.keys()
    results["detected"] = detected_lesions.values()
    results["area"] = lesion_sizes

    return results



def compute_dice_from_pred(predictions_path):

    split = predictions_path.split("/")[-1].split("_")[1]

    dice = Dice()

    predictions = torch.load(predictions_path, map_location="cpu")
    ground_truth = torch.load(f"{path.join(*predictions_path.split('/')[:-1])}/groundtruth_{split}.pt", map_location = "cpu")

    dice_score = dice(predictions, ground_truth)

    return dice_score


def compute_dice_per_pred(predictions_path, threshold = None):

    split = predictions_path.split("/")[-1].split("_")[1]

    dice = Dice()

    dices = []

    predictions = torch.load(predictions_path, map_location="cpu")
    ground_truth = torch.load(f"{path.join(*predictions_path.split('/')[:-1])}/groundtruth_{split}.pt", map_location = "cpu")

    for i in range(predictions.shape[0]):

        if threshold is not None:
            predictions[i][predictions[i] > threshold] = 1
            predictions[i][predictions[i] <= threshold] = 0
            
        dice_score = dice(predictions[i], ground_truth[i]).item()
        dices.append(dice_score)

    return dices





def compute_dice(model, split: str, gpu = "cuda:1"):
    data_path = f"/home/u887755/internship/256/data_256/Lung/axial/Preprocessed_data/{split}"
    data = DataLoaderSegmentation(data_path, create_three_channels=True)
    dataloader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=False)
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")

    dice = Dice()

    model.eval()
    model = model.to(device)

    predictions = []
    ground_truth = []


    # make predictions
    for batch in tqdm(dataloader):
        # break
        img, mask = batch
        img = img.to(device)
        mask = mask.to(device)


        with torch.no_grad():
            output = torch.sigmoid(model(img))

        predictions.append(output)
        ground_truth.append(mask)

    predictions = torch.cat(predictions)
    ground_truth = torch.cat(ground_truth)

    dice_score = dice(predictions, ground_truth)

    return dice_score


def compute_hausdorff_from_pred(predictions_path, return_mean = True):

    split = predictions_path.split("/")[-1].split("_")[1]



    predictions = torch.load(predictions_path, map_location="cpu")
    ground_truth = torch.load(f"{path.join(*predictions_path.split('/')[:-1])}/groundtruth_{split}.pt", map_location = "cpu")


    ground_truth[ground_truth >0.5] = 1
    ground_truth[ground_truth <=0.5] = 0
    predictions[predictions > 0.5] = 1
    predictions[predictions <= 0.5] = 0

    predictions = predictions.numpy()
    ground_truth = ground_truth.numpy()

    ground_truth = ground_truth[~np.all(predictions == 0, axis=(2, 3))]
    predictions = predictions[~np.all(predictions == 0, axis=(2, 3))]


    ground_truth = torch.tensor(ground_truth)
    predictions = torch.tensor(predictions)

    ground_truth = ground_truth.unsqueeze(1)
    predictions = predictions.unsqueeze(1)

    hausdorffs = compute_hausdorff_distance(ground_truth, predictions, directed=True)
    # documentation says compute_hausdorff_distance(pred, gt), but this does not give the correct result
    # That would give the directed distance from the prediction to the ground truth but is should be the other way around
    
    return np.mean(np.array(hausdorffs)) if return_mean else hausdorffs


def save_predictions(model, split, savepath, gpu = "cuda:1"):
    data_path = f"/home/u887755/internship/256/data_256/Lung/axial/Preprocessed_data/{split}"
    data = DataLoaderSegmentation(data_path, create_three_channels=True)
    dataloader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=False)
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")

    model.eval()
    model = model.to(device)

    predictions = []
    ground_truth = []


    # make predictions
    for batch in tqdm(dataloader):
        # break
        img, mask = batch
        img = img.to(device)
        mask = mask.to(device)


        with torch.no_grad():
            output = torch.sigmoid(model(img))

        predictions.append(output)
        ground_truth.append(mask)

    predictions = torch.cat(predictions)
    ground_truth = torch.cat(ground_truth)

    torch.save(predictions, savepath)

def best_model_path(model_folder, metric = "loss"):
    if model_folder[-1] == "/":
        model_folder = model_folder[:-1]
    best_losses = glob.glob(f"{model_folder}/checkpoints/*_{metric}_*")
    best_losses.sort(key=lambda x: int(x.split("/")[-1].split("_")[1]))
    print(best_losses[-1])
    return best_losses[-1]

def load_pretrained_swin(model_folder, criterion = "dice"):
    model = SwinTransformerV2(img_size = 256, num_classes=1, drop_rate=0.2, window_size=8, depths = [2,2,2,2])
    model.load_state_dict(torch.load(best_model_path(model_folder, criterion)))
    return model




if __name__ == "__main__":

    val_100_1p_pred = "evaluation/predictions/distance100_val_1p.pt"
    test_100_1p_pred = "evaluation/predictions/distance100_test_1p.pt"

    val_100_10p_pred = "evaluation/predictions/distance100_val_10p.pt"
    test_100_10p_pred = "evaluation/predictions/distance100_test_10p.pt"

    print("1p")
    print("validation")
    print("Dice", compute_dice_from_pred(val_100_1p_pred).item())
    print("Hausdorff", compute_hausdorff_from_pred(val_100_1p_pred).item())
    print("Detection", np.mean(lesion_detection_from_predictions(val_100_1p_pred)["detected"]))
    print("\n")

    print("test")
    print("Dice", compute_dice_from_pred(test_100_1p_pred))
    print("Hausdorff", compute_hausdorff_from_pred(test_100_1p_pred))
    print("Detection", np.mean(lesion_detection_from_predictions(test_100_1p_pred)["detected"]))

    print("\n")
    print("\n")


    print("10p")
    print("validation")
    print("Dice", compute_dice_from_pred(val_100_10p_pred))
    print("Hausdorff", compute_hausdorff_from_pred(val_100_10p_pred))
    print("Detection", np.mean(lesion_detection_from_predictions(val_100_10p_pred)["detected"]))
    print("\n")

    print("test")
    print("Dice", compute_dice_from_pred(test_100_10p_pred))
    print("Hausdorff", compute_hausdorff_from_pred(test_100_10p_pred))
    print("Detection", np.mean(lesion_detection_from_predictions(test_100_10p_pred)["detected"]))



    # print(compute_dice(load_pretrained_swin("trained_models/swin_unetv2/Lung/distance/1percent/swin_unetv2_1percent_100patients", "dice"), "val"))
    # print(compute_dice(load_pretrained_swin("trained_models/swin_unetv2/Lung/distance/1percent/swin_unetv2_1percent_100patients", "dice"), "test"))


    # model = SwinTransformerV2(img_size = 256, num_classes=1, drop_rate=0.2, window_size=8, depths = [2,2,2,2])
    # model.load_state_dict(torch.load("trained_models/swin_unetv2/Lung/distance/swin_unetv2_full/checkpoints/epoch_914_best_dice_0.634.pth"))
    # data_path = "../internship/256/data_256/Lung/axial/Preprocessed_data/val"

    # n_detected, n_non_detected = detection_rate(model, data_path)



    pass
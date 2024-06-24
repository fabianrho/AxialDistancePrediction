import sys

sys.path.append(".")
sys.path.append("..")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import torch
import os.path as path
from utils.data_loading import DataLoaderSegmentation
import numpy as np

from skimage import measure

from utils.metrics import Dice

def plot_comparison(split = "val", index = [0], crop_size = 64, x_shift = 0, y_shift = 0,save_folder = "plots", save_id = "", evaluation_folder = "predictions_2", data_percentage = "100p", methods = ["ImageNet", "Distance", "DCL", "BYOL", "SimMIM"]):



    if type(index) == int:
        index = [index]

    if type(crop_size) == int:
        crop_size = [crop_size]
        crop_sizes = crop_size * len(index)
    else:
        crop_sizes = crop_size

    if type(x_shift) == int:
        x_shifts = [x_shift] * len(index)
    else:
        x_shifts = x_shift
    
    if type(y_shift) == int:
        y_shifts = [y_shift] * len(index)
    else:
        y_shifts = y_shift
        

    fig, axs = plt.subplots(7, len(index), figsize=(len(index)*8, len(index)*12))

    for j, ind in enumerate(index):

        x_shift = x_shifts[j]
        y_shift = y_shifts[j]

        crop_size = crop_sizes[j]


        data_path = f"../../internship/256/data_256/Lung/axial/Preprocessed_data/{split}"
        data = DataLoaderSegmentation(data_path, create_three_channels=True)

        ground_truth_for_dice = torch.load(f"{evaluation_folder}/groundtruth_{split}.pt", map_location = "cpu")[ind]
        ground_truth = ground_truth_for_dice.cpu().squeeze(0).numpy()

        region_props = measure.regionprops(measure.label(ground_truth))

        if len(region_props) > 1:
            print("Multiple regions found on ", ind)


        centroid = np.array(region_props[0].centroid)
        shifted_centroid = centroid + np.array([y_shift,x_shift])
        shifted_centroid_plot = centroid + np.array([x_shift,y_shift])
        ground_truth  = ground_truth[int(shifted_centroid[0])-crop_size:int(shifted_centroid[0])+crop_size, int(shifted_centroid[1])-crop_size:int(shifted_centroid[1])+crop_size]
        # ground_truth  = ground_truth[int(centroid[0])-crop_size+x_shift:int(centroid[0])+crop_size+x_shift, int(centroid[1])-crop_size+y_shift:int(centroid[1])+crop_size+y_shift]

        image = data[ind][0][0].numpy()




        dice = Dice()

        # ground_truth = np.ma.masked_where(ground_truth == 0, ground_truth)


        dice_score = {}
        predictions = {}
        for method in methods:
            pred = torch.load(f"{evaluation_folder}/{method.lower()}_{split}_{data_percentage}.pt", map_location = "cpu")[ind]


            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0
            dice_score[method] = dice(pred, ground_truth_for_dice)


            pred = pred.cpu().squeeze(0).numpy()


            # crop around centroid
            pred = pred[int(shifted_centroid[0])-crop_size:int(shifted_centroid[0])+crop_size, int(shifted_centroid[1])-crop_size:int(shifted_centroid[1])+crop_size]

            # pred = np.ma.masked_where(pred == 0, pred)


            predictions[method] = pred


        # axs = axs.flatten()

        anchor_point = (int(shifted_centroid[0])-crop_size, int(256-shifted_centroid[1]-crop_size))
        axs[0,j].plot(*anchor_point, 'ro')

        rect = plt.Rectangle(anchor_point, crop_size*2, crop_size*2, edgecolor='r', facecolor='none', lw=4, angle = 360)
        axs[0,j].add_patch(rect)

        axs[0,j].imshow(np.rot90(image), cmap="gray")
        # axs[0].imshow(np.ma.masked_where(ground_truth == 0, ground_truth), cmap=ListedColormap(['red']))
        axs[0,j].set_title("Scan Patch", fontsize = 44)
        axs[0,j].axis("off")

        image_min, image_max = np.min(image), np.max(image)

        image = image[int(shifted_centroid[0])-crop_size:int(shifted_centroid[0])+crop_size, int(shifted_centroid[1])-crop_size:int(shifted_centroid[1])+crop_size]

        axs[1,j].imshow(np.rot90(image), cmap="gray", vmin= image_min, vmax=image_max)
        axs[1,j].set_title("Cropped Lesion", fontsize = 44)

        
        axs[1,j].axis("off")


        for i, method in enumerate(methods):


            axs[i+2,j].imshow(np.rot90(image), cmap="gray", vmin= image_min, vmax=image_max)
            axs[i+2,j].imshow(np.rot90(np.ma.masked_where(ground_truth == 0, ground_truth)), cmap=ListedColormap(['red']))
            axs[i+2,j].imshow(np.rot90(np.ma.masked_where(predictions[method] == 0, predictions[method])), cmap=ListedColormap(['green']))
            false_positives = np.logical_and(predictions[method] == 1, ground_truth == 0)
            # print(np.sum(false_positives))
            
            axs[i+2,j].imshow(np.rot90(np.ma.masked_where(false_positives == 0, false_positives)), cmap=ListedColormap(['yellow']))
            method_title = method
            if method == "Distance":
                method_title = "ADL"
            axs[i+2,j].set_title(method_title + f" ({dice_score[method]:.3f})", fontsize=44)
            axs[i+2,j].axis("off")
            # rotate the image

        

    plt.savefig(f"{save_folder}/{split}_{data_percentage}_{save_id}.png", bbox_inches='tight', dpi = 150)
    plt.show()

    plt.close()
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings


# take in an arbitrary number of files and plot the data
def plot_history(savefolders: list, legend_labels: list = [], metric: str = "Dice", split: str = "Validation", ewm: bool = False, alpha: float = None, max_epoch: int = None, save_path: str = None):

    if len(legend_labels) != 0 and len(legend_labels) != len(savefolders):
        raise ValueError("legend_labels must be the same length as savefolders")
    if len(legend_labels) == 0:
        legend_labels = [savefolder.split("/")[-1] for savefolder in savefolders]

    if not ewm and alpha is not None:
        warnings.warn("alpha is only used when ewm is enabled", UserWarning)

    message = []

    plt.figure(figsize=(6.4,4.8))


    for i, folder in enumerate(savefolders):
        try:
            history = pd.read_csv(os.path.join(folder, "training_log.csv"))
            # remove duplicate epochs
            history = history.drop_duplicates(subset="Epoch", keep="last")
            if max_epoch is not None:
                history = history[history["Epoch"] <= max_epoch]
            # if history['Epoch'].max() != 500:
            message.append(f"{legend_labels[i]}: {round(history[f'{split} {metric}'].max(),3)}, (ep {history[f'{split} {metric}'].idxmax()+1} / {history['Epoch'].max()})")
            # else:
                # message.append(f"{legend_labels[i]}: {round(history[f'{split} {metric}'].max(),3)}, (ep {history[f'{split} {metric}'].idxmax()+1})")

            if ewm:
                if alpha is None:
                    alpha = 0.1
                history[f"{split} {metric}"] = history[f"{split} {metric}"].ewm(alpha=alpha).mean()

            plt.plot(history["Epoch"], history[f"{split} {metric}"], label=f"{legend_labels[i]}")
        except Exception as e:
            print(e)
            print(f"Could not find {folder}/training_log.csv")
            continue

        
    plt.xlabel("Epoch")
    plt.ylabel(f"{split} {metric}")
    plt.legend()
    # legend title
    plt.legend(title="Model (pretraining)")
    if ewm:
        plt.suptitle(f"{split} {metric}")
        plt.title(f"(Exponential Weighted Moving Average, $\\alpha = {{{alpha}}}$)", fontsize=10)

    else:
        plt.title(f"{split} {metric}")
    if max_epoch is not None:
        plt.xlim(0,max_epoch)

    # plt.ylim(0,1)
    # plt.show()
    if save_path is not None:
        plt.savefig(save_path, dpi = 600, bbox_inches='tight')
    else:
        plt.show()
    return message
    


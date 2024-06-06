import numpy as np
import regex as re
import pandas as pd
import matplotlib.pyplot as plt


def curve_from_slurm(slurm_path):
    with open(slurm_path, 'r') as f:
        lines = f.readlines()
    lines = [l for l in lines if 'Epoch' in l]

    epochs = []
    losses = []

    for l in lines:
        epoch = int(re.findall(r'Epoch (\d+)', l)[0])
        loss = float(re.findall(r'Val Loss: (\d+\.\d+)', l)[0])
        epochs.append(epoch)
        losses.append(loss)

    return np.array(epochs), np.array(losses)

if __name__ == '__main__':
    epochs, losses = curve_from_slurm("slurm/distancerib.cinereous.8896.out")
    history = pd.DataFrame({'epoch': epochs, 'loss': losses})
    plt.plot(history['epoch'], history['loss'])
    plt.savefig("temp/curve.png")
import imageio
import os
import matplotlib.pyplot as plt

folder = "trained_models/unetr_pelvis/validation_visualisation"

filenames = sorted(os.listdir(folder))

# sort by number in filename
filenames.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))


images = []
for i, filename in enumerate(filenames):
    img = imageio.imread(f"{folder}/{filename}")
    plt.imshow(img)

    images.append(img)
imageio.mimsave('traininggif.gif', images)
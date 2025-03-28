import numpy as np
import pickle as pickle
import os

folder = '/Users/tolmach/Documents/GitHub/ActivationMattersRNN/data/fixed_points'
for subfolder in os.listdir(folder):
    if subfolder == '.DS_Store':
        continue
    path = os.path.join(folder, subfolder)
    for file in os.listdir(path):
        if file == '.DS_Store':
            continue
        shuffle = file.split("_")[2].split("=")[1] == 'True'
        if shuffle:
            extra = "controlType=shuffled_control=True"
        else:
            extra = "control=False"
        pieces = file.split(f"shuffle={shuffle}")
        new_file = pieces[0] + extra
        for i in range(1, len(pieces)):
            new_file += pieces[i]
        print(new_file)
        os.rename(os.path.join(path, file), os.path.join(path, new_file))
import numpy as np
import os
import json
import pickle

path = "/Users/tolmach/Documents/GitHub/ActivationMattersRNN/data/latent_circuits/"
folders = os.listdir(path)
folders = [folder for folder in folders if folder != ".DS_Store"]
taskname = "CDDM"
LC_data = {}
for folder in folders:
    task = folder.split("_")[0]
    if taskname != task:
        pass
    else:
        activation_function = folder.split("_")[1]
        constrained = folder.split("_")[2].split("=")[1]
        sfolders = os.listdir(os.path.join(path, folder))
        sfolders = [sfolder for sfolder in sfolders if sfolder != ".DS_Store"]
        if not (activation_function in LC_data.keys()):
            LC_data[activation_function] = {}
        if not (constrained in LC_data[activation_function].keys()):
            LC_data[activation_function][f"Dale={constrained}"] = {}
        for sfolder in sfolders:
            ssfolders = os.listdir(os.path.join(path, folder, sfolder))
            ssfolders = [ssfolder for ssfolder in ssfolders if ssfolder != ".DS_Store"]
            scores = []
            for ssfolder in ssfolders:
                scores.append(float(ssfolder.split("_")[0]))
            scores = np.array(scores)
            top_ssfolder = ssfolders[np.argmax(scores)]
            if not (sfolder in LC_data[activation_function][f"Dale={constrained}"].keys()):
                LC_data[activation_function][f"Dale={constrained}"][sfolder] = {}
            for file in os.listdir(os.path.join(path, folder, sfolder, top_ssfolder)):
                if "LC_params" in file:
                    break
            LC_file = os.path.join(path, folder, sfolder, top_ssfolder, file)
            LC_data[activation_function][f"Dale={constrained}"][sfolder]["LC_params"] = json.load(open(LC_file, "rb+"))

            for file in os.listdir(os.path.join(path, folder, sfolder, top_ssfolder)):
                if "RNN_params" in file:
                    break
            RNN_file = os.path.join(path, folder, sfolder, top_ssfolder, file)
            LC_data[activation_function][f"Dale={constrained}"][sfolder]["RNN_params"] = json.load(open(RNN_file, "rb+"))
pickle.dump(LC_data, open(os.path.join(path, f"LC_data_of_top_10RNNs_{taskname}.pkl"), "wb+"))


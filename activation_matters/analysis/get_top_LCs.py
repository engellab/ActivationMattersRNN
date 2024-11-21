import numpy as np
import os
import json
import pickle
import yaml
from tqdm.auto import tqdm
path = "/Users/tolmach/Documents/GitHub/latent_circuit_inference/data/inferred_LCs/"
folders = os.listdir(path)
folders = [folder for folder in folders if folder != ".DS_Store"]
taskname = "CDDM"
LC_data = {}

def check_constrained(W_inp):
    W_inp = np.array(W_inp)
    W_inp_diag = np.zeros_like(W_inp)
    for i in range(W_inp.shape[1]):
        W_inp_diag[i, i] = W_inp[i, i]
    W_inp_no_diag = W_inp - W_inp_diag
    return (np.max(W_inp_no_diag) < 2e-3)



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

            # for each ssfolder go in and check the files. You need to filter the ssfolders out
            ssfolder_filtered = []
            for ssfolder in tqdm(ssfolders):
                path_to_params_file = os.path.join(path, folder, sfolder, ssfolder, "LC_params.json")
                if os.path.exists(path_to_params_file):
                    with open(path_to_params_file, "r") as file:
                        params = json.load(file)
                    if check_constrained(params["W_inp"]):
                        ssfolder_filtered.append(ssfolder)


                # with open(os.path.join(path, folder, sfolder, ssfolder, "full_LC_config.yaml"), "r") as file:
                #     cfg = yaml.safe_load(file)
                # if cfg.get("lambda_behavior", -1) == 0.25:
                #     print(ssfolder)

            scores = []
            for ssfolder in ssfolder_filtered:
                scores.append(float(ssfolder.split("_")[0]))
            scores = np.array(scores)

            top_ssfolder = ssfolder_filtered[np.argmax(scores)]
            print(folder, sfolder, top_ssfolder)
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


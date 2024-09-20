import os
import numpy as np
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
import json
import yaml

def get_traces(N, dt, tau, W_inp, W_rec, W_out, activation_name, activation_slope, task):
    RNN = RNN_numpy(N=N,
                    dt=dt, tau=tau,
                    W_inp=W_inp,
                    W_rec=W_rec,
                    W_out=W_out,
                    activation_name=activation_name,
                    activation_slope=activation_slope)

    inputs, targets, conditions = task.get_batch()
    RNN.clear_history()
    RNN.sigma_rec = 0
    RNN.sigma_inp = 0
    RNN.y = np.zeros(RNN.N)
    RNN.run(inputs, sigma_rec=0, sigma_inp=0)
    traces = RNN.get_history()
    return traces, conditions

def get_subfolders(path, pattern):
    dirs = os.listdir(path)
    return [dir for dir in dirs if pattern in dir and os.path.isdir(os.path.join(path, dir))]

def collected_relevant_folders(trained_RNNs_folder, taskname):
    task_folders = get_subfolders(trained_RNNs_folder, taskname)
    RNNs_folders = []
    for folder in task_folders:
        rnn_folders = os.listdir(os.path.join(trained_RNNs_folder, folder))
        RNNs_folders.extend([os.path.join(folder, rnn_folder) for rnn_folder in rnn_folders])
    RNNs_folders = [folder for folder in RNNs_folders if not (".DS_Store" in folder)]
    return RNNs_folders

def load_data(folder):
    files = os.listdir(folder)
    for file in files:
        if "_params_" in file:
            return json.load(open(os.path.join(folder, file), 'rb+'))

def get_RNN_score(folder):
    files = os.listdir(folder)
    for file in files:
        if "_params_" in file:
            return float(file.split("_")[0])

def load_config(folder):
    files = os.listdir(folder)
    for file in files:
        if "config" in file:
            if "json" in file:
                config_file = json.load(open(os.path.join(folder, file), 'rb+'))
            elif "yaml" in file:
                config_file = yaml.safe_load(open(os.path.join(folder, file), 'rb+'))
            else:
                raise("The config file has wrong extention")
            return config_file
    raise ("There is no config file!")


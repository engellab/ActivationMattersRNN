from copy import deepcopy

from activation_matters.utils.feautre_extraction_utils import get_dataset
from trainRNNbrain.training.training_utils import *
import numpy as np
np.set_printoptions(suppress=True)
import pickle
import pandas as pd
from tqdm.auto import tqdm
import hydra
from omegaconf import OmegaConf
from activation_matters.utils.make_RNN_datasets_utils import *

taskname = 'CDDM'
print(taskname)
os.environ['HYDRA_FULL_ERROR'] = '1'
OmegaConf.register_new_resolver("eval", eval)
@hydra.main(version_base="1.3", config_path=f"../../configs/task", config_name=f'{taskname}')
def make_dataset(cfg):
    RNNs_folders = collected_relevant_folders(cfg.task.paths.trained_RNNs_path, taskname)

    # defining the task
    task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.task.dt)
    task = hydra.utils.instantiate(task_conf)
    keys = ["folder",
            "W_inp_RNN", "W_rec_RNN", "W_out_RNN",
            "dt", "tau", "N",
            "sigma_inp_RNN", "sigma_rec_RNN",
            "constrained", "n_steps", "lr_RNN", "mask",
            "lambda_orth", "lambda_r", "orth_input_only",
            "RNN_score", "RNN_maxiter",
            "activation_name", "activation_slope",
            "RNN_trajectories", "conditions",
            "weight_decay", "same_batch", "p"]
    data_dict = {}
    for key in keys:
        data_dict[key] = []

    for RNN_folder in tqdm(RNNs_folders):
        RNN_folder_path = os.path.join(cfg.task.paths.trained_RNNs_path, RNN_folder)
        RNN_data = load_data(RNN_folder_path)
        RNN_score = get_RNN_score(RNN_folder_path)
        W_out = np.array(RNN_data["W_out"])
        W_inp = np.array(RNN_data["W_inp"])
        W_rec = np.array(RNN_data["W_rec"])

        RNN_config = load_config(RNN_folder_path)

        # if it's an old config style - pass
        if not ("model" in RNN_config.keys()):
            print(RNN_folder)
            continue
            # dt = RNN_config["dt"]
            # tau = RNN_config["tau"]
            # p = RNN_config.get("p", 2)
            # constrained = RNN_config["constrained"]
            # activation_name = RNN_config["activation"]
            # sigma_inp_RNN = RNN_config["sigma_inp"]
            # sigma_rec_RNN = RNN_config["sigma_rec"]
            # if taskname == "DMTS":
            #     RNN_maxiter = RNN_config["max_iter_1"]
            #     lambda_r = RNN_config["lambda_r_1"]
            # if RNN_maxiter < 100:
            #     print(f"Skipped: {RNN_folder_path}. You should delete it!")
            #     continue
            # n_steps = RNN_config["n_steps"]
            # lr_RNN = RNN_config["lr"]
            # lambda_orth = RNN_config["lambda_orth"]
            # weight_decay = RNN_config["weight_decay"]
            # same_batch = RNN_config["same_batch"]
            # orth_input_only = RNN_config["orth_input_only"]
            # activation_slope = RNN_config.get("activation_slope", 1)
        else:
            dt = RNN_config["model"]["dt"]
            tau = RNN_config["model"]["tau"]
            p = RNN_config["trainer"]["p"]
            constrained = RNN_config["model"]["constrained"]
            activation_name = RNN_config["model"]["activation_name"]
            sigma_inp_RNN = RNN_config["model"]["sigma_inp"]
            sigma_rec_RNN = RNN_config["model"]["sigma_rec"]
            RNN_maxiter = RNN_config["trainer"]["max_iter"]
            lambda_r = RNN_config["trainer"]["lambda_r"]

            if taskname == "DMTS" or taskname == 'MemoryNumber':
                RNN_maxiter = RNN_config["trainer"]["max_iter"][-1]
                lambda_r = RNN_config["trainer"]["lambda_r"][-1]
            if RNN_maxiter < 100:
                print(f"Skipped: {RNN_folder_path}. You should delete it!")
                continue
            n_steps = RNN_config["task"]["T"]/RNN_config["model"]["dt"]
            lr_RNN = RNN_config["trainer"]["lr"]
            lambda_orth = RNN_config["trainer"]["lambda_orth"]
            weight_decay = RNN_config["trainer"]["weight_decay"]
            same_batch = RNN_config["trainer"]["same_batch"]
            mask = get_training_mask(OmegaConf.create(RNN_config["task"]), dt=RNN_config["model"]["dt"])
            orth_input_only = RNN_config["trainer"]["orth_input_only"]
            activation_slope = RNN_config["model"].get("activation_slope", 1)
        RNN_trajectories, conditions = get_traces(N=W_rec.shape[0], dt=dt, tau=tau,
                                                  W_inp=W_inp,
                                                  W_rec=W_rec,
                                                  W_out=W_out,
                                                  activation_name=activation_name,
                                                  activation_slope=activation_slope,
                                                  task=task)
        data_dict["folder"].append(np.copy(RNN_folder))
        data_dict["RNN_trajectories"].append(np.copy(RNN_trajectories))
        data_dict["conditions"].append(np.copy(conditions))
        data_dict["RNN_score"].append(RNN_score)
        data_dict["activation_name"].append(activation_name)
        data_dict["activation_slope"].append(activation_slope)
        data_dict["dt"].append(dt)
        data_dict["tau"].append(tau)
        data_dict["W_inp_RNN"].append(np.copy(np.array(W_inp)))
        data_dict["W_rec_RNN"].append(np.copy(np.array(W_rec)))
        data_dict["W_out_RNN"].append(np.copy(np.array(W_out)))
        data_dict["N"].append(W_rec.shape[0])
        data_dict["sigma_inp_RNN"].append(sigma_inp_RNN)
        data_dict["sigma_rec_RNN"].append(sigma_rec_RNN)
        data_dict["n_steps"].append(n_steps)
        data_dict["lr_RNN"].append(lr_RNN)
        data_dict["mask"].append(mask)
        data_dict["lambda_orth"].append(lambda_orth)
        data_dict["lambda_r"].append(lambda_r)
        data_dict["orth_input_only"].append(orth_input_only)
        data_dict["RNN_maxiter"].append(RNN_maxiter)
        data_dict["constrained"].append(constrained)
        data_dict["weight_decay"].append(weight_decay)
        data_dict["same_batch"].append(same_batch)
        data_dict["p"].append(p)
    df = pd.DataFrame(data_dict)
    df.sort_values(by="RNN_score", inplace=True)
    print(df["RNN_score"])
    pickle.dump(df, open(os.path.join(cfg.task.paths.RNN_datasets_path, f"{taskname}.pkl"), 'wb+'))

    # save top 30 nets
    n_nets = 30
    file_str = os.path.join(cfg.task.paths.RNN_datasets_path, f"{taskname}.pkl")
    DF_dict = {}
    activations_list = ["relu", "sigmoid", "tanh"]
    constrained_list = [True, False]
    for activation_name in activations_list:
        DF_dict[activation_name] = {}
        RNN_score = eval(f"cfg.task.dataset_filtering_params.{activation_name}_filters.RNN_score_filter")
        lambda_r = eval(f"cfg.task.dataset_filtering_params.{activation_name}_filters.lambda_r")
        activation_slope = eval(f"cfg.task.dataset_filtering_params.{activation_name}_filters.activation_slope")
        for constrained in constrained_list:
            filters = {"activation_name": ("==", activation_name),
                       "activation_slope": ("==", activation_slope),
                       "RNN_score": ("<=", RNN_score),
                       "constrained": ("==", constrained),
                       "lambda_r": (">=", lambda_r)}
            dataset = get_dataset(file_str, filters)
            print(f"{activation_name};Dale={constrained}", len(dataset))
            DF_dict[activation_name][f"Dale={constrained}"] = deepcopy(dataset[:n_nets])
    pickle.dump(DF_dict, open(os.path.join(cfg.task.paths.RNN_datasets_path, f"{taskname}_top{n_nets}.pkl"), 'wb+'))


if __name__ == '__main__':
    make_dataset()
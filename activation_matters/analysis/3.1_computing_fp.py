from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from trainRNNbrain.training.training_utils import prepare_task_arguments
from activation_matters.utils.trajectories_utils import shuffle_connectivity
import os
import pickle
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.analyzers.DynamicSystemAnalyzer import *
from tqdm.auto import tqdm
import time
import hydra
np.set_printoptions(suppress=True)
OmegaConf.register_new_resolver("eval", eval)

taskname = "CDDM"
show = False
save = True
@hydra.main(version_base="1.3", config_path=f"../../configs/task", config_name=f'{taskname}')
def calc_fp(cfg):
    taskname = cfg.task.taskname
    dataset_path = os.path.join(f"{cfg.task.paths.dataset_path}", f"{taskname}_top30.pkl")
    dataset = pickle.load(open(dataset_path, "rb"))
    data_save_folder = cfg.task.paths.fixed_points_data_folder

    # defining the task
    task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.task.dt)
    task = hydra.utils.instantiate(task_conf)
    if hasattr(task, 'random_window'):
        task.random_window = 0 # no randomness in the task structure for the analysis

    connectivity_dict = {}
    for activation_name in ["relu", "sigmoid", "tanh"]:
        for constrained in [True, False]:
            connectivity_dict[activation_name] = {}
            connectivity_dict[activation_name]["inp"] = dataset[activation_name][f"Dale={constrained}"]["W_inp_RNN"].tolist()
            connectivity_dict[activation_name]["rec"] = dataset[activation_name][f"Dale={constrained}"]["W_rec_RNN"].tolist()
            connectivity_dict[activation_name]["out"] = dataset[activation_name][f"Dale={constrained}"]["W_out_RNN"].tolist()
            n_nets = len(dataset[activation_name][f"Dale={constrained}"])
            for shuffle in [False, True]:
                print(f"{activation_name};Dale={constrained};shuffled={shuffle}", len(dataset[activation_name][f"Dale={constrained}"]))
                for i in tqdm(range(n_nets)):
                    W_inp, W_rec, W_out = (connectivity_dict[activation_name][tp][i] for tp in ["inp", "rec", "out"])
                    if shuffle:
                        W_inp, W_rec, W_out = shuffle_connectivity(W_inp, W_rec, W_out)

                    N = W_inp.shape[0]
                    # assumes that all the RNNs of the same type have the same activation slope
                    activation_slope = dataset[activation_name][f"Dale={constrained}"]["activation_slope"].tolist()[0]

                    net_params = {"N": N,
                                  "dt": cfg.task.dt,
                                  "tau": cfg.task.tau,
                                  "activation_name": activation_name,
                                  "activation_slope": activation_slope,
                                  "W_inp": W_inp,
                                  "W_rec": W_rec,
                                  "W_out": W_out,
                                  "bias_rec": None,
                                  "y_init": np.zeros(N)}
                    rnn = RNN_numpy(**net_params)

                    # quantify the fixed points
                    dsa = DynamicSystemAnalyzer(rnn, task)

                    inputs = cfg.task.dynamical_topology_analysis.inputs
                    t0 = time.time()
                    for input in inputs:
                        dsa.get_fixed_points(Input=np.array(input),
                                             **cfg.task.dynamical_topology_analysis.fp_search_params)
                    t1 = time.time()
                    print(f"Executed in : {np.round(t1-t0, 2)} seconds")

                    fixed_points = []
                    labels = []
                    for k, input_as_key in enumerate(list(dsa.fp_data.keys())):
                        if "stable_fps" in dsa.fp_data[input_as_key].keys():
                            sfp = dsa.fp_data[input_as_key]['stable_fps']
                            fixed_points.append(sfp)
                            labels.extend([f"sfp_{k}"] * sfp.shape[0])
                        if "unstable_fps" in dsa.fp_data[input_as_key].keys():
                            ufp = dsa.fp_data[input_as_key]['unstable_fps']
                            labels.extend([f"ufp_{k}"] * ufp.shape[0])
                            fixed_points.append(ufp)
                    try:
                        fixed_points = np.vstack(fixed_points)
                    except:
                        print("No fixed points!")
                        fixed_points = [] # no fixed points were found!
                    fp_data_processed = {"fps": fixed_points, "labels": labels}

                    if not os.path.exists(data_save_folder):
                        Path(data_save_folder).mkdir(parents=True, exist_ok=False)

                    file_path = os.path.join(data_save_folder,
                                             f"{activation_name}_constrained={constrained}_shuffle={shuffle}_fpstruct_{i}.pkl")
                    pickle.dump(obj=fp_data_processed, file=open(file_path, "wb+"))

if __name__ == '__main__':
    calc_fp()
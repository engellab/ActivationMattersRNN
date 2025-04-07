from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from trainRNNbrain.rnns import RNN_torch
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
# OmegaConf.register_new_resolver("eval", eval)
import ray

# Define the remote function to process each network
@ray.remote
def process_network(i,
                    activation_name,
                    constrained,
                    control_type,
                    control,
                    dataSegment,
                    data_save_folder,
                    connectivity_dict,
                    dataset,
                    cfg,
                    task):
    if control:
        file_path = os.path.join(data_save_folder,
                                 f"{activation_name}_constrained={constrained}_controlType={control_type}_control={control}_fpstruct_{dataSegment}_n={i}.pkl")
    else:
        file_path = os.path.join(data_save_folder,
                                 f"{activation_name}_constrained={constrained}_control={control}_fpstruct_{dataSegment}_n={i}.pkl")


    if not os.path.isfile(file_path):
        # Extract connectivity matrices
        if control == False:
            W_inp, W_rec, W_out = (connectivity_dict[activation_name][tp][i] for tp in ["inp", "rec", "out"])
        else:
            W_inp, W_rec, W_out = (connectivity_dict[activation_name][tp][i] for tp in ["inp", "rec", "out"])
            if control_type == 'shuffled':
                W_inp, W_rec, W_out = shuffle_connectivity(W_inp, W_rec, W_out)
            elif control_type == 'random':
                if constrained:
                    W_rec, W_inp, W_out, _, _, _, _ = RNN_torch.get_connectivity_Dale(N=W_inp.shape[0],
                                                                                      num_inputs=task.n_inputs,
                                                                                      num_outputs=task.n_outputs)
                else:
                    W_rec, W_inp, W_out, _, _, _ = RNN_torch.get_connectivity(N=W_inp.shape[0],
                                                                              num_inputs=task.n_inputs,
                                                                              num_outputs=task.n_outputs)
                W_inp = W_inp.detach().cpu().numpy()
                W_rec = W_rec.detach().cpu().numpy()
                W_out = W_out.detach().cpu().numpy()

        N = W_inp.shape[0]
        # Assumes that all the RNNs of the same type have the same activation slope
        activation_slope = dataset[activation_name][f"Dale={constrained}"]["activation_slope"].tolist()[0]

        # Define network parameters
        net_params = {
            "N": N,
            "dt": cfg.task.dt,
            "tau": cfg.task.tau,
            "activation_name": activation_name,
            "activation_slope": activation_slope,
            "W_inp": W_inp,
            "W_rec": W_rec,
            "W_out": W_out,
            "bias_rec": None,
            "y_init": np.zeros(N)
        }

        # Initialize and run the RNN
        rnn = RNN_numpy(**net_params)

        # Quantify the fixed points
        dsa = DynamicSystemAnalyzer(rnn, task)

        inputs = cfg.task.dynamical_topology_analysis.inputs
        t0 = time.time()
        for input in inputs:
            dsa.get_fixed_points(Input=np.array(input),
                                 **cfg.task.dynamical_topology_analysis.fp_search_params)
        t1 = time.time()
        print(f"Executed in : {np.round(t1 - t0, 2)} seconds")

        # Process fixed points
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
            fixed_points = []  # No fixed points were found!

        # Save the processed data
        fp_data_processed = {"fps": fixed_points, "labels": labels}
        with open(file_path, "wb+") as f:
            pickle.dump(fp_data_processed, f)
    return None


show = False
save = True
@hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'base')
def computing_fp(cfg):
    os.environ["NUMEXPR_MAX_THREADS"] = "50"
    os.environ["RAY_DEDUP_LOGS"] = "0"
    n_nets = cfg.n_nets
    dataSegment = cfg.dataSegment
    taskname = cfg.task.taskname
    dataset_path = os.path.join(f"{cfg.paths.RNN_dataset_path}", f"{taskname}_{dataSegment}{n_nets}.pkl")
    dataset = pickle.load(open(dataset_path, "rb"))
    data_save_folder = os.path.join(cfg.paths.fixed_points_data_folder, taskname)
    control_type = cfg.control_type

    # defining the task
    task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.task.dt)
    task = hydra.utils.instantiate(task_conf)
    if hasattr(task, 'random_window'):
        task.random_window = 0 # no randomness in the task structure for the analysis

    ray.init(ignore_reinit_error=True, address="auto")
    print(ray.available_resources())

    connectivity_dict = {}
    for activation_name in ["relu", "sigmoid", "tanh"]:
        for constrained in [True, False]:
            connectivity_dict[activation_name] = {}
            connectivity_dict[activation_name]["inp"] = dataset[activation_name][f"Dale={constrained}"]["W_inp_RNN"].tolist()
            connectivity_dict[activation_name]["rec"] = dataset[activation_name][f"Dale={constrained}"]["W_rec_RNN"].tolist()
            connectivity_dict[activation_name]["out"] = dataset[activation_name][f"Dale={constrained}"]["W_out_RNN"].tolist()
            for control in [False, True]:
                print(f"{activation_name};Dale={constrained};control_type={control_type}; control={control}",
                      len(dataset[activation_name][f"Dale={constrained}"]))


                # Ensure the data save folder exists
                if not os.path.exists(data_save_folder):
                    Path(data_save_folder).mkdir(parents=True, exist_ok=True)

                results = [process_network.remote(i,
                                                  activation_name,
                                                  constrained,
                                                  control_type,
                                                  control,
                                                  dataSegment,
                                                  data_save_folder,
                                                  connectivity_dict,
                                                  dataset,
                                                  cfg,
                                                  task)
                           for i in range(n_nets)]

                # Retrieve results as they become available
                for res in tqdm(results):
                    ray.get(res)  # Wait for each task to complete
    ray.shutdown()
    return None

if __name__ == '__main__':
    computing_fp()
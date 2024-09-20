from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from trainRNNbrain.training.training_utils import prepare_task_arguments
from activation_matters.plots.ploting_utils import plot_fixed_points
from activation_matters.utils.feautre_extraction_utils import get_dataset
from activation_matters.utils.trajectories_utils import shuffle_connectivity
from style.style_setup import set_up_plotting_styles
import os
import pickle
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.analyzers.DynamicSystemAnalyzer import *
from tqdm.auto import tqdm
import time
import hydra
np.set_printoptions(suppress=True)
os.system('python ../../style/style_setup.py')

n_nets = 30
show = False
save = True
OmegaConf.register_new_resolver("eval", eval)
@hydra.main(version_base="1.3", config_path=f"../../configs/task", config_name=f'MemoryNumber')
def calc_fp(cfg):
    taskname = cfg.task.taskname
    set_up_plotting_styles(cfg.task.paths.style_path)
    file_str = os.path.join(cfg.task.paths.RNN_datasets_path, f"{taskname}.pkl")
    data_save_folder = cfg.task.paths.fixed_points_data_folder
    img_save_folder = cfg.task.paths.img_folder

    # defining the task
    task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.task.dt)
    task = hydra.utils.instantiate(task_conf)
    if hasattr(task, 'random_window'):
        task.random_window = 0
    DF_dict = {}
    connectivity_dict = {}
    for activation_name in ["relu", "sigmoid", "tanh"]:
        for constrained in [True, False]:
            activation_slope = eval(f"cfg.task.dataset_filtering_params.{activation_name}_filters.activation_slope")
            filters = {"activation_name": ("==", activation_name),
                       "activation_slope": ("==", activation_slope),
                       "RNN_score": ("<=", eval(f"cfg.task.dataset_filtering_params.{activation_name}_filters.RNN_score_filter")),
                       "constrained": ("==", constrained),
                       "lambda_r": (">=", eval(f"cfg.task.dataset_filtering_params.{activation_name}_filters.lambda_r")),
                       "n_steps": ("==", eval(f"cfg.task.dataset_filtering_params.{activation_name}_filters.n_steps"))}

            DF_dict[activation_name] = get_dataset(file_str, filters)[:n_nets]
            connectivity_dict[activation_name] = {}
            for tp in ["inp", "rec", "out"]:
                connectivity_dict[activation_name][tp] = DF_dict[activation_name][f"W_{tp}_RNN"].tolist()

            for shuffle in [False, True]:
                print(f"Activation {activation_name}; constrained={constrained}; shuffle={shuffle}")
                for i in tqdm(range(n_nets)):
                    W_inp, W_rec, W_out = (connectivity_dict[activation_name][tp][i] for tp in ["inp", "rec", "out"])
                    if shuffle:
                        W_inp, W_rec, W_out = shuffle_connectivity(W_inp, W_rec, W_out)

                    N = W_inp.shape[0]
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
                    # file_path_load = os.path.join(data_save_folder,
                    #                          f"{activation_name}_constrained={constrained}_fpstruct_{i}.pkl")
                    # fp_data_processed = pickle.load(file=open(file_path_load, "rb+"))


                    file_path = os.path.join(data_save_folder,
                                             f"{activation_name}_constrained={constrained}_shuffle={shuffle}_fpstruct_{i}.pkl")
                    pickle.dump(obj=fp_data_processed, file=open(file_path, "wb+"))

                    # colors = [list(cfg.task.dynamical_topology_analysis.colors[k]) for k in range(len(dsa.fp_data.keys()))]
                    # if OmegaConf.select(cfg.task.dynamical_topology_analysis, "markers") is not None:
                    #     markers = [list(cfg.task.dynamical_topology_analysis.markers[k]) for k in range(len(dsa.fp_data.keys()))]
                    # else:
                    #     markers = None
                    # if OmegaConf.select(cfg.task.dynamical_topology_analysis, "edgecolors") is None or cfg.task.dynamical_topology_analysis.edgecolors == 'None':
                    #     edgecolors = None
                    # else:
                    #     edgecolors = [list(cfg.task.dynamical_topology_analysis.edgecolors[k]) for k in
                    #                   range(len(dsa.fp_data.keys()))]
                    # path = os.path.join(img_save_folder, f"fp_struct_{activation_name}_constrained={constrained}_net={i}.pdf")
                    # plot_fixed_points(fixed_point_struct=fixed_points, fp_labels=labels,
                    #                   colors=colors,
                    #                   markers=markers,
                    #                   edgecolors=edgecolors,
                    #                   n_dim=2, show=show, save=save, path=path)
                    # path = os.path.join(img_save_folder, f"fp_struct3D_{activation_name}_constrained={constrained}_net={i}.pdf")
                    # plot_fixed_points(fixed_point_struct=fixed_points, fp_labels=labels,
                    #                   colors=colors,
                    #                   markers=markers,
                    #                   edgecolors=edgecolors,
                    #                   n_dim=3, show=show, save=save, path=path)

if __name__ == '__main__':
    calc_fp()
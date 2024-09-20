from DSA import DSA
import numpy as np
from activation_matters.plots.ploting_utils import plot_embedding, plot_similarity_matrix
from sklearn.manifold import MDS
from trainRNNbrain.training.training_utils import *
from activation_matters.utils.trajectories_utils import *
from style.style_setup import set_up_plotting_styles
from itertools import chain
from copy import deepcopy
import hydra
from omegaconf import OmegaConf
np.set_printoptions(suppress=True)

OmegaConf.register_new_resolver("eval", eval)
taskname = "MemoryNumber"
show = True
save = True

@hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'{taskname}_conf')
def dynamic_similarity_analysis(cfg):
    n_nets = cfg.DSA_params.n_nets
    n_delays = cfg.DSA_params.n_delays
    delay_interval = 1
    rank = cfg.DSA_params.rank
    img_save_folder = cfg.paths.img_folder
    set_up_plotting_styles(cfg.paths.style_path)
    save = cfg.trajectory_analysis_params.save_figures
    # defining the task
    task_conf = prepare_task_arguments(cfg_task=cfg, dt=cfg.dt)
    task = hydra.utils.instantiate(task_conf)
    if hasattr(task, "random_window"):
        task.random_window = 0
    task.seed = 0 # for consistency
    get_batch_args = {}
    if taskname == "DMTS":
        get_batch_args = {"num_rep": 4}

    DF_dict = {}
    file_str = os.path.join(cfg.paths.RNN_datasets_path, f"{taskname}.pkl")
    activations_list = ["relu", "sigmoid", "tanh"]
    constrained_list = [True, False]
    RNN_trajectories = []
    legends = []
    inds_list = []
    cnt = 0
    for activation_name in activations_list:
        DF_dict[activation_name] = {}
        for constrained in constrained_list:
            RNN_score = eval(f"cfg.dataset_filtering_params.{activation_name}_filters.RNN_score_filter")
            lambda_r = eval(f"cfg.dataset_filtering_params.{activation_name}_filters.lambda_r")
            activation_slope = eval(f"cfg.dataset_filtering_params.{activation_name}_filters.activation_slope")
            filters = {"activation_name": ("==", activation_name),
                       "activation_slope": ("==", activation_slope),
                       "RNN_score": ("<=", RNN_score),
                       "constrained": ("==", constrained),
                       "lambda_r": (">=", lambda_r)}
            dataset = get_dataset(file_str, filters)
            print(f"{activation_name};Dale={constrained}", len(dataset))
            DF_dict[activation_name][f"Dale={constrained}"] = deepcopy(dataset[:n_nets])

            dkeys = (f"{activation_name}", f"Dale={constrained}")
            legends.append(f"{activation_name} Dale={constrained}")
            trajectories = get_trajectories(dataset=DF_dict[dkeys[0]][dkeys[1]],
                                           task=task,
                                           activation_name=activation_name,
                                           activation_slope=activation_slope,
                                           get_batch_args=get_batch_args)
            RNN_trajectories.append(trajectories)
            inds_list.append(cnt + np.arange(len(trajectories)))
            cnt += len(trajectories)
    RNN_trajectories = list(chain(*RNN_trajectories))

    # GET THE SIMILARITY BETWEEN THE TRAJECTORIES
    print("Calculating the similarity between the trajectories")
    RNN_trajectories = [RNN_trajectories.swapaxes(0, -1).astype(np.float32) for RNN_trajectories in RNN_trajectories]

    dsa = DSA(RNN_trajectories,
              n_delays=n_delays,
              rank=rank,
              delay_interval=delay_interval,
              verbose=True,
              device='cpu',
              iters=1000,
              lr=1e-2)
    Mat = dsa.fit_score()
    path = os.path.join(img_save_folder, f"DSA_similarity_matrix.pdf")
    plot_similarity_matrix(Mat, save=save, path=path)

    colors = ["red", "orange", "blue", "deepskyblue", "green", "lightgreen"]
    for attempt in range(5):
        mds = MDS(n_components=2)
        mds.fit(Mat)
        embedding = mds.embedding_
        path = os.path.join(img_save_folder, f"MDS_DSA_attempt={attempt}.pdf")
        plot_embedding(embedding, inds_list, legends, colors, save=save, path=path, show=show)


if __name__ == '__main__':
    dynamic_similarity_analysis()
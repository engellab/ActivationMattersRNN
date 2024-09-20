import sys
import numpy as np
np.set_printoptions(suppress=True)
from trainRNNbrain.training.training_utils import *
import hydra
from omegaconf import OmegaConf
# from style.style_setup import set_up_plotting_styles
from activation_matters.utils.trajectories_utils import *
from itertools import chain
from copy import deepcopy
OmegaConf.register_new_resolver("eval", eval)
show = False
save = True

def get_trajectory_similarity(feature_list):
    n_dim = feature_list[0].shape[0]
    Mat = np.zeros((len(feature_list), len(feature_list)))
    for i in tqdm(range(len(feature_list))):
        for j in range(i + 1, len(feature_list)):
            F1 = feature_list[i].reshape(n_dim, -1).T
            F2 = feature_list[j].reshape(n_dim, -1).T
            M1, residuals, rank, s = np.linalg.lstsq(F1, F2, rcond=None)
            score1 = np.sqrt(np.sum((F1 @ M1 - F2) ** 2))
            M2, residuals, rank, s = np.linalg.lstsq(F2, F1, rcond=None)
            score2 = np.sqrt(np.sum((F2 @ M2 - F1) ** 2))
            Mat[i, j] = Mat[j, i] = (score1 + score2) / 2
    return Mat

def project_trajectories(trajectory, n_dim=10):
    pca = PCA(n_components=n_dim)
    trajectories_flattened = trajectory.reshape(trajectory.shape[0], -1)
    N = trajectory.shape[0]
    T = trajectory.shape[1]
    K = trajectory.shape[2]
    res = pca.fit_transform(trajectories_flattened.T).T.reshape(n_dim, T, K)
    print(f"Explained variance by {n_dim} PCs: {np.sum(pca.explained_variance_ratio_)}")
    return res

# @hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'base')
@hydra.main(version_base="1.3", config_path=f"../../configs/task", config_name=f'CDDM')
def analysis_of_trajectories(cfg):
    taskname = cfg.task.taskname
    # set_up_plotting_styles(cfg.task.paths.style_path)
    n_nets = cfg.task.trajectory_analysis_params.n_nets
    aux_datasets_folder = cfg.task.paths.auxilliary_datasets_path
    n_PCs = cfg.task.trajectory_analysis_params.n_PCs

    # defining the task
    task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.task.dt)
    task = hydra.utils.instantiate(task_conf)
    if taskname == "CDDM":
        task.coherences = np.array(list(cfg.task.trajectory_analysis_params.coherences))
    if hasattr(task, 'random_window'):
        task.random_window = 0
    task.seed = 0 # for consistency

    DF_dict = {}
    file_str = os.path.join(cfg.task.paths.RNN_datasets_path, f"{taskname}.pkl")

    activations_list = ["relu", "sigmoid", "tanh"]
    constrained_list = [True, False]
    shuffle_list = [False, True]  #
    RNN_trajectories = []
    legends = []
    inds_list = []
    cnt = 0
    for activation_name in activations_list:
        DF_dict[activation_name] = {}
        for constrained in constrained_list:
            RNN_score = eval(f"cfg.task.dataset_filtering_params.{activation_name}_filters.RNN_score_filter")
            lambda_r = eval(f"cfg.task.dataset_filtering_params.{activation_name}_filters.lambda_r")
            activation_slope = eval(f"cfg.task.dataset_filtering_params.{activation_name}_filters.activation_slope")
            filters = {"activation_name": ("==", activation_name),
                       "activation_slope": ("==", activation_slope),
                       "RNN_score": ("<=", RNN_score),
                       "constrained": ("==", constrained),
                       "lambda_r": (">=", lambda_r)}
            dataset = get_dataset(file_str, filters)
            print(f"{activation_name};Dale={constrained}", len(dataset))
            DF_dict[activation_name][f"Dale={constrained}"] = deepcopy(dataset[:n_nets])

            for shuffle in shuffle_list:
                dkeys = (f"{activation_name}", f"Dale={constrained}", f"shuffle={shuffle}")
                legends.append(f"{activation_name} Dale={constrained} shuffle={shuffle}")

                get_traj = get_trajectories_shuffled_connectivity if shuffle == True else get_trajectories
                trajectories = get_traj(dataset=DF_dict[dkeys[0]][dkeys[1]],
                               task=task,
                               activation_name=activation_name,
                               activation_slope=activation_slope,
                               get_batch_args={})
                RNN_trajectories.append(trajectories)
                inds_list.append(cnt + np.arange(len(trajectories)))
                cnt += len(trajectories)

    RNN_trajectories = list(chain.from_iterable(RNN_trajectories))
    RNN_trajectories_projected = []
    for i in tqdm(range(len(RNN_trajectories))):
        projected_trajectory = project_trajectories(RNN_trajectories[i], n_dim=n_PCs)

        # divide by overall variance
        R = np.sqrt(np.sum(np.var(projected_trajectory.reshape(projected_trajectory.shape[0], -1), axis = 1)))
        projected_trajectory_normalized = projected_trajectory / R

        RNN_trajectories_projected.append(projected_trajectory_normalized)

    data_dict = {}
    data_dict["RNN_trajectories"] = RNN_trajectories
    data_dict["RNN_trajectories_projected"] = RNN_trajectories_projected
    data_dict["legends"] = legends
    data_dict["inds_list"] = inds_list
    pickle.dump(data_dict, open(os.path.join(aux_datasets_folder, "trajectories_data.pkl"), "wb"))

    # GET THE SIMILARITY BETWEEN THE TRAJECTORIES
    print("Calculating the similarity between the trajectories")
    Mat = get_trajectory_similarity(RNN_trajectories_projected)
    pickle.dump(Mat, open(os.path.join(aux_datasets_folder, "trajectories_similarity_matrix.pkl"), "wb"))

if __name__ == '__main__':
    analysis_of_trajectories()
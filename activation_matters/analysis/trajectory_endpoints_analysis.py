import numpy as np
from trainRNNbrain.training.training_utils import prepare_task_arguments
from activation_matters.utils.feautre_extraction_utils import get_dataset
from activation_matters.plots.ploting_utils import interpolate_color
from activation_matters.utils.trajectories_utils import shuffle_connectivity
from scipy.linalg import orthogonal_procrustes
np.set_printoptions(suppress=True)
import os
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.analyzers.DynamicSystemAnalyzer import *
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from scipy.optimize import minimize
import pickle
import hydra
import ray
from omegaconf import OmegaConf
# OmegaConf.register_new_resolver("eval", eval)


@ray.remote
def computing_trajectory_endpoints_similarity_inner_loop(i, j, Pi, Pj):
    Q1, _ = orthogonal_procrustes(Pi, Pj)
    score1 = np.sqrt(np.sum((Pi @ Q1 - Pj) ** 2))
    Q2, _ = orthogonal_procrustes(Pj, Pi)
    score2 = np.sqrt(np.sum((Pj @ Q2 - Pi) ** 2))
    return (i, j), (score1 + score2) / 2

@ray.remote
def process_network(i, activation_name, connectivity_dict, shuffle, dataset, constrained, cfg, inputs, n_PCs):
    # Extract connectivity matrices
    W_inp, W_rec, W_out = (connectivity_dict[activation_name][tp][i] for tp in ["inp", "rec", "out"])

    # Shuffle connectivity if required
    if shuffle:
        W_inp, W_rec, W_out = shuffle_connectivity(W_inp, W_rec, W_out)

    # Extract activation slope
    activation_slope = dataset[activation_name][f"Dale={constrained}"]["activation_slope"].tolist()[0]
    N = W_inp.shape[0]

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
    rnn.run(inputs)
    trajectories = rnn.get_history()
    stim_representations = trajectories[:, -1, :]

    # Perform PCA
    pca = PCA(n_components=n_PCs)
    X = pca.fit_transform(stim_representations.T)
    print(f"Explained variance by {n_PCs} PCs: {np.sum(pca.explained_variance_ratio_)}")

    # Normalize (remove the scaling difference)
    R = np.sqrt(np.mean(np.sum((X - np.mean(X, axis=0)) ** 2, axis=1)))
    X = X / R
    return X

def get_trajectory_endpoints_similarity(PCA_stimuli_list):
    Mat = np.zeros((len(PCA_stimuli_list), len(PCA_stimuli_list)))
    elements = []
    for i in tqdm(range(len(PCA_stimuli_list))):
        row = []
        for j in range(i + 1, len(PCA_stimuli_list)):
            Pi = PCA_stimuli_list[i]
            Pj = PCA_stimuli_list[j]
            row.append(computing_trajectory_endpoints_similarity_inner_loop.remote(i, j, Pi, Pj))
        elements.extend(ray.get(row))

    for element in elements:
        inds = element[0]
        score = element[1]
        Mat[inds] = Mat[inds[::-1]] = score
    return Mat


show = True
save = False
@hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'base')
def trajectory_endpoints_analysis(cfg):
    n_nets = cfg.n_nets
    dataSegment = cfg.dataSegment
    taskname = cfg.task.taskname
    dataset_path = os.path.join(f"{cfg.paths.RNN_dataset_path}", f"{taskname}_{dataSegment}{n_nets}.pkl")
    aux_datasets_folder = os.path.join(f"{cfg.paths.auxilliary_datasets_path}", taskname)
    dataset = pickle.load(open(dataset_path, "rb"))
    file_path = os.path.join(aux_datasets_folder, f"trajectory_endpoints_similarity_{dataSegment}{n_nets}.pkl")

    ray.init(ignore_reinit_error=True)

    if not os.path.exists(file_path):
        n_PCs = cfg.task.trajectory_endpoints_analysis_params.n_PCs
        connectivity_dict = {}
        # defining the task
        task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.task.dt)
        task = hydra.utils.instantiate(task_conf)
        if taskname == "CDDM":
            task.coherences = np.array(list(cfg.task.trajectory_endpoints_analysis_params.coherences))
        if hasattr(task, 'random_window'):
            task.random_window = 0
        task.seed = 0 # for consistency
        get_batch_args = {}
        inputs, targets, conditions = task.get_batch(**get_batch_args)
        legends = []
        inds_list = []
        PCA_stimuli_list = []
        cnt = 0
        for activation_name in ["relu", "sigmoid", "tanh"]:
            for constrained in [True, False]:
                connectivity_dict[activation_name] = {}
                connectivity_dict[activation_name]["inp"] = dataset[activation_name][f"Dale={constrained}"]["W_inp_RNN"].tolist()
                connectivity_dict[activation_name]["rec"] = dataset[activation_name][f"Dale={constrained}"]["W_rec_RNN"].tolist()
                connectivity_dict[activation_name]["out"] = dataset[activation_name][f"Dale={constrained}"]["W_out_RNN"].tolist()
                for shuffle in [False, True]:
                    print(f"Activation {activation_name}; constrained={constrained}; shuffle={shuffle}")
                    # Launch tasks in parallel
                    results = [process_network.remote(i, activation_name,
                                                      connectivity_dict,
                                                      shuffle,
                                                      dataset,
                                                      constrained,
                                                      cfg,
                                                      inputs,
                                                      n_PCs)
                               for i in range(n_nets)]

                    # Retrieve results as they become available
                    for res in tqdm(results):
                        PCA_stimuli_list.append(ray.get(res))

                    # Update indices and legends
                    inds_list.append(cnt + np.arange(n_nets))
                    cnt += n_nets
                    legends.append(f"{activation_name} Dale={constrained} shuffle={shuffle}")

        data_dict = {}
        data_dict["PCA_stimuli_list"] = PCA_stimuli_list
        data_dict["inds_list"] = inds_list
        data_dict["legends"] = legends

        if taskname == "CDDM":
            face_colors, edge_colors, markers = get_plotting_params_CDDM(conditions)
        elif taskname == "GoNoGo" or taskname == "MemoryNumber":
            face_colors, edge_colors, markers = get_plotting_params_GoNoGo(conditions)

        data_dict["face_colors"] = face_colors
        data_dict["edge_colors"] = edge_colors
        data_dict["markers"] = markers
        pickle.dump(data_dict, open(file_path, "wb+"))

    data_dict = pickle.load(open(file_path, "rb+"))
    file_path = os.path.join(aux_datasets_folder, f"trajectory_endpoints_similarity_matrix_{dataSegment}{n_nets}.pkl")
    if not os.path.exists(file_path):
        # GET THE SIMILARITY BETWEEN THE TRAJECTORIES
        PCA_stimuli_list = data_dict["PCA_stimuli_list"]
        print("Calculating the similarity between the stimuli representations")
        Mat = get_trajectory_endpoints_similarity(PCA_stimuli_list)
        pickle.dump(Mat, open(file_path, "wb+"))
    ray.shutdown()
    return None


def get_plotting_params_CDDM(conditions):
    contexts = np.array([1 if conditions[i]['context'] == 'motion' else -1 for i in range(len(conditions))])
    relevant_coherences = [conditions[i]["color_coh"]
                           if conditions[i]["context"] == "color"
                           else conditions[i]["motion_coh"] for i in range(len(conditions))]
    irrelevant_coherences = [conditions[i]["motion_coh"]
                             if conditions[i]["context"] == "color"
                             else conditions[i]["color_coh"] for i in range(len(conditions))]
    primary_colors = np.array([[0.3, 0.4, 0.8],
                               [0.8, 0.8, 0.8],
                               [0.8, 0.2, 0.3]])

    face_colors = []
    low_val = np.min(relevant_coherences)
    high_val = np.max(relevant_coherences)
    mid_val = np.mean(relevant_coherences)
    for coherence in relevant_coherences:
        color = interpolate_color(low_color=primary_colors[0],
                                  mid_color=primary_colors[1],
                                  high_color=primary_colors[2],
                                  low_val=low_val,
                                  mid_val=mid_val,
                                  high_val=high_val,
                                  val=coherence)
        face_colors.append(color)

    edge_colors = []
    low_val = np.min(irrelevant_coherences)
    high_val = np.max(irrelevant_coherences)
    mid_val = np.mean(irrelevant_coherences)
    for coherence in irrelevant_coherences:
        color = interpolate_color(low_color=primary_colors[0],
                                  mid_color=primary_colors[1],
                                  high_color=primary_colors[2],
                                  low_val=low_val,
                                  mid_val=mid_val,
                                  high_val=high_val,
                                  val=coherence)
        edge_colors.append(color)
    markers = ["o" if context == 1 else "p" for context in contexts]
    return face_colors, edge_colors, markers

def get_plotting_params_GoNoGo(conditions):
    face_colors = []
    primary_colors = np.array([[0.3, 0.4, 0.8],
                               [0.8, 0.8, 0.8],
                               [0.8, 0.2, 0.3]])
    values = [float(str(condition).split(",")[0].split(":")[1]) for condition in conditions]
    low_val = np.min(values)
    high_val = np.max(values)
    mid_val = np.mean(values)
    for value in values:
        color = interpolate_color(low_color=primary_colors[0],
                                  mid_color=primary_colors[1],
                                  high_color=primary_colors[2],
                                  low_val=low_val,
                                  mid_val=mid_val,
                                  high_val=high_val,
                                  val=value)
        face_colors.append(color)
    edge_colors = [np.array([0, 0, 0]) for val in values]
    markers = ["o" for val in values]
    return face_colors, edge_colors, markers

if __name__ == '__main__':
    trajectory_endpoints_analysis()
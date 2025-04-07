from itertools import chain

import numpy as np
from trainRNNbrain.training.training_utils import prepare_task_arguments
from activation_matters.utils.feautre_extraction_utils import get_dataset
from activation_matters.plots.ploting_utils import interpolate_color
from activation_matters.utils.trajectories_utils import *
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
def extract_feature(trajectory, n_PCs):
    stim_representations = trajectory[:, -1, :]

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
feature_type = "trajectory_endpoints"
@hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'base')
def trajectory_endpoints_analysis(cfg):
    os.environ["NUMEXPR_MAX_THREADS"] = "50"
    n_nets = cfg.n_nets
    dataSegment = cfg.dataSegment
    taskname = cfg.task.taskname
    dataset_path = os.path.join(f"{cfg.paths.RNN_dataset_path}", f"{taskname}_{dataSegment}{n_nets}.pkl")
    aux_datasets_folder = os.path.join(f"{cfg.paths.auxilliary_datasets_path}", taskname)
    dataset = pickle.load(open(dataset_path, "rb"))
    n_PCs = cfg.task.trajectory_endpoints_analysis_params.n_PCs
    control_type = cfg.control_type #shuffled or untrained

    file_path = os.path.join(aux_datasets_folder, f"{feature_type}_{dataSegment}{n_nets}_{control_type}.pkl")
    if not os.path.exists(file_path):

        #printing RNN scores for top n_nets networks
        for activation in dataset.keys():
            for constraint in dataset[activation].keys():
                scores = dataset[activation][constraint]["R2_score"]
                m = np.mean(scores)
                s = np.std(scores)
                print(f"{activation}, {constraint} R2 score = {np.round(m, 5)} +- {np.round(s, 5)}")

        # defining the task
        task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.task.dt)
        task = hydra.utils.instantiate(task_conf)
        if taskname == "CDDM":
            task.coherences = np.array(list(cfg.task.trajectory_analysis_params.coherences))
        if hasattr(task, 'random_window'):
            task.random_window = 0 # eliminating any source of randomness while analysing the trajectories
        task.seed = 0 # for consistency

        activations_list = ["relu", "sigmoid", "tanh"]
        constrained_list = [True, False]
        control_list = [False, True]
        RNN_features = []
        legends = []
        inds_list = []
        cnt = 0

        for activation_name in activations_list:
            for constrained in constrained_list:
                for control in control_list:
                    if control:
                        print(f"{activation_name};Dale={constrained};control={control_type}",
                              len(dataset[activation_name][f"Dale={constrained}"]))
                        legends.append(f"{activation_name} Dale={constrained} {control_type}={control}")
                        if control_type == 'shuffled':
                            shuffled = True
                            random = False
                        elif control_type == 'random':
                            shuffled = False
                            random = True
                    else:
                        print(f"{activation_name};Dale={constrained}",
                              len(dataset[activation_name][f"Dale={constrained}"]))
                        legends.append(f"{activation_name} Dale={constrained}")
                        shuffled = False
                        random = False

                    # assumes that all the RNNs of the same type have the same activation slope
                    activation_slope = dataset[activation_name][f"Dale={constrained}"]["activation_slope"].tolist()[0]
                    trajectories = get_trajectories(dataset=dataset[activation_name][f"Dale={constrained}"],
                                                    task=task,
                                                    activation_name=activation_name,
                                                    activation_slope=activation_slope,
                                                    get_batch_args={},
                                                    shuffled=shuffled,
                                                    random=random)
                    RNN_features.append(trajectories)
                    inds_list.append(cnt + np.arange(len(trajectories)))
                    cnt += len(trajectories)

        RNN_features = list(chain.from_iterable(RNN_features))
        RNN_features_processed = []

        # Launch tasks in parallel
        ray.init(ignore_reinit_error=True, address="auto")
        print(ray.available_resources())

        results = [extract_feature.remote(feature, n_PCs) for feature in RNN_features]
        for res in tqdm(results):
            RNN_features_processed.append(ray.get(res))
        ray.shutdown()

        data_dict = {}
        data_dict[f"RNN_{feature_type}"] = RNN_features
        data_dict[f"RNN_{feature_type}_processed"] = RNN_features_processed
        data_dict["legends"] = legends
        data_dict["inds_list"] = inds_list
        pickle.dump(data_dict, open(file_path, "wb+"))

    data_dict = pickle.load(open(file_path, "rb+"))
    file_path = os.path.join(aux_datasets_folder, f"{feature_type}_similarity_matrix_{dataSegment}{n_nets}_{control_type}.pkl")

    # GET THE SIMILARITY BETWEEN THE TRAJECTORIES
    if not os.path.exists(file_path):
        print(f"Calculating the similarity between the {feature_type}")
        RNN_features_processed = data_dict[f"RNN_{feature_type}_processed"]
        Mat = get_trajectory_endpoints_similarity(RNN_features_processed)
        pickle.dump(Mat, open(file_path, "wb+"))


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
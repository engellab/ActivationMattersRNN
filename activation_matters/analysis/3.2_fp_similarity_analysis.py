import os
from scipy.linalg import orthogonal_procrustes
from scipy.stats import ortho_group
from sklearn.decomposition import PCA
from copy import copy
import pickle
from activation_matters.plots.ploting_utils import normalize_color
import numpy as np
import hydra
from omegaconf import OmegaConf
from tqdm.auto import tqdm
OmegaConf.register_new_resolver("eval", eval)
from matplotlib import pyplot as plt
def ICP_registration(points_source, labels_source,
                     points_target, labels_target,
                     n_tries=51, max_iter=1000, tol=1e-10,
                     method="procrustes"):
    Qs = []
    mses = []
    for tries in range(n_tries):
        i = 0
        change = np.inf
        mse_prev = np.inf
        mse = np.inf
        Q = ortho_group.rvs(dim=points_target.shape[1])
        Q = Q[:points_source.shape[1], :points_target.shape[1]]
        while i < max_iter and change > tol:
            # for each source point, find a matching target point
            D = get_distance_matrix(points_source @ Q, labels_source,
                                    points_target, labels_target)
            # clean this matrix of all the rows not being assigned
            mask = ~np.isnan(D).all(axis=1)
            points_source_filtered = points_source[mask, :]
            D = D[mask, :]
            points_target_matched = points_target[np.nanargmin(D, axis=1), :]
            if method == 'procrustes':
                Q, _ = orthogonal_procrustes(points_source_filtered, points_target_matched)
            if method == "regression":
                Q, _, _, _ = np.linalg.lstsq(points_source_filtered, points_target_matched, rcond=None)
            mse = np.sqrt(np.sum((points_source_filtered @ Q - points_target_matched) ** 2))
            change = (mse_prev - mse)
            mse_prev = mse
            i += 1
        Qs.append(np.copy(Q))
        mses.append(np.copy(mse))
    best_mse = np.min(mses)
    best_Q = Qs[np.argmin(mses)]

    # source = points_source_filtered @ best_Q
    # target = points_target_matched
    # fig = plt.figure()
    # plt.scatter(source[:, 0], source[:, 1])
    # plt.scatter(target[:, 0], target[:, 1])
    # plt.show()

    return best_Q, best_mse

def get_distance_matrix(points_source, labels_source, points_target, labels_target):
    types = []
    types.extend(labels_source)
    types.extend(labels_target)
    types = np.unique(types)
    distance = np.inf * np.ones((len(points_source), len(points_target)))
    for type in types:
        inds_source = np.where(labels_source == type)[0]
        inds_target = np.where(labels_target == type)[0]
        if len(inds_source) > 0 and len(inds_target) > 0:
            points_of_type_source = np.repeat(points_source[inds_source, np.newaxis, :], repeats=len(inds_target), axis=1)
            points_of_type_target = np.repeat(points_target[np.newaxis, inds_target, :], repeats=len(inds_source), axis=0)
            distance[np.ix_(inds_source, inds_target)] = np.linalg.norm(points_of_type_source - points_of_type_target, axis=2)
    return distance

def get_fp_similarity(fp_dict_combined, method='procrustes'):
    fp_list = fp_dict_combined["fp_list"]
    labels_list = fp_dict_combined["labels_list"]
    Mat = np.zeros((len(fp_list), len(fp_list)))
    for i in tqdm(range(len(fp_list))):
        for j in (range(i + 1, len(fp_list))):
            _, score_ij = ICP_registration(points_source=fp_list[j], labels_source=labels_list[j],
                                        points_target=fp_list[i], labels_target=labels_list[i], method=method)
            _, score_ji = ICP_registration(points_source=fp_list[i], labels_source=labels_list[i],
                                        points_target=fp_list[j], labels_target=labels_list[j], method=method)
            Mat[i, j] = Mat[j, i] = (score_ij + score_ji) / 2
    return Mat

taskname = "CDDM"
n_dim = 2
n_nets_MDS = 30
save = True
show = True
normalized = True #whether to normalize the fixed point configuration scale or not
# @hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'base')
@hydra.main(version_base="1.3", config_path=f"../../configs/task", config_name=f'{taskname}')
def fixed_point_registration(cfg):
    path_to_folder = cfg.task.paths.fixed_points_data_folder
    aux_datasets_folder = cfg.task.paths.auxilliary_datasets_path
    n_PCs = cfg.task.dynamical_topology_analysis.n_PCs
    net_types = ["relu", "sigmoid", "tanh"]
    data_dict = get_fp_data_dict(path_to_folder, net_types, n_PCs, normalized)
    fp_dict_combined = {}
    fp_dict_combined["fp_list"] = []
    fp_dict_combined["labels_list"] = []
    legends = []
    inds_list = []
    cnt = 0

    # combine the fixed points together in one dataset
    for net_type in net_types:
        for constrained in [True, False]:
            for shuffle in [False, True]:
                fp_dict = data_dict[net_type][f"constrained={constrained}"][f"shuffle={shuffle}"]
                n_nets = len(fp_dict["fp_list"])
                n_nets_taken = min([n_nets_MDS, n_nets])
                fp_dict_combined["fp_list"].extend([fp_dict["fp_list"][i] for i in range(n_nets_taken)])
                fp_dict_combined["labels_list"].extend([fp_dict["labels_list"][i] for i in range(n_nets_taken)])
                inds_list.append(cnt + np.arange(n_nets_taken))
                cnt += n_nets_taken
                legends.append(f"{net_type}_constrained={constrained}_shuffle={shuffle}")
    fp_dict_combined["inds_list"] = inds_list
    fp_dict_combined["legends"] = legends
    pickle.dump(fp_dict_combined, open(os.path.join(aux_datasets_folder, f"FP_similarity_normalized={normalized}.pkl"), "wb+"))

    Mat = get_fp_similarity(fp_dict_combined, method='procrustes')
    print(Mat)
    pickle.dump(Mat, open(os.path.join(aux_datasets_folder, f"FP_similarity_matrix_normalized={normalized}.pkl"), "wb"))

def get_fp_data_dict(path_to_folder, net_types, n_PCs, normalized=False):
    folder = os.listdir(path_to_folder)
    data_dict = {}
    for net_type in net_types:
        data_dict[net_type] = {}
        for constrained in [True, False]:
            data_dict[net_type][f"constrained={constrained}"] = {}
            for shuffle in [False, True]:
                data_dict[net_type][f"constrained={constrained}"][f"shuffle={shuffle}"] = {}
                data_dict[net_type][f"constrained={constrained}"][f"shuffle={shuffle}"]["fp_list"] = []
                data_dict[net_type][f"constrained={constrained}"][f"shuffle={shuffle}"]["labels_list"] = []
                data_dict[net_type][f"constrained={constrained}"][f"shuffle={shuffle}"]["fp_dict_list"] = []

    for file in folder:
        if file == ".DS_Store" or file == "fp old":
            pass
        else:
            net_type = file.split("_")[0]
            constrained = eval(file.split("_")[1].split("=")[1])
            shuffle = eval(file.split("_")[2].split("=")[1])
            path = os.path.join(path_to_folder, file)
            opened_file = open(path, "rb+")
            fixed_points_dict = pickle.load(opened_file)
            fixed_points = fixed_points_dict["fps"]
            labels = fixed_points_dict["labels"]

            if type(fixed_points) == list: # it sometimes happens that if shuffled network there are not fixed points
                F = np.zeros((1, n_PCs))
            else:
                pca = PCA(n_components=n_PCs)
                F = pca.fit_transform(fixed_points)
                if normalized:
                    F = (F - np.mean(F, axis=0))
                    R = np.sqrt(np.mean(np.sum(F ** 2, axis=1)))
                    F = F / R # remove the scale
                    print(f"{net_type}_constrained={constrained}_shuffle={shuffle}; Explained variance by {n_PCs} PCs: {np.sum(pca.explained_variance_ratio_)} R:{R}")
                else:
                    print(f"{net_type}_constrained={constrained}_shuffle={shuffle}; Explained variance by {n_PCs} PCs: {np.sum(pca.explained_variance_ratio_)}")
            # print(f"{net_type}_constrained={constrained}_shuffle={shuffle}; Explained variance by {n_PCs} PCs: {np.sum(pca.explained_variance_ratio_)}")
            d = {}
            input_IDs = np.unique(np.array([int(l.split("_")[-1]) for l in labels]))
            for i in input_IDs:
                for point_type in ["sfp", "ufp"]:
                    if f"{point_type}_{i}" in np.unique(labels):
                        inds = np.where(np.array(labels) == f"{point_type}_{i}")
                        d[f"{point_type}_{i}"] = F[inds]
                    else:
                        d[f"{point_type}_{i}"] = np.array([])

            data_dict[net_type][f"constrained={constrained}"][f"shuffle={shuffle}"]["fp_list"].append(np.copy(F))
            data_dict[net_type][f"constrained={constrained}"][f"shuffle={shuffle}"]["labels_list"].append(np.copy(labels))
            data_dict[net_type][f"constrained={constrained}"][f"shuffle={shuffle}"]["fp_dict_list"].append(copy(d))
    return data_dict

if __name__ == '__main__':
    fixed_point_registration()
import itertools
import os
from scipy.linalg import orthogonal_procrustes
from scipy.stats import ortho_group
from sklearn.decomposition import PCA
from copy import copy
import pickle
import numpy as np
import hydra
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import ray
import time
import asyncio

# Ray remote function for parallel computation
@ray.remote(num_cpus=1)
def computing_fixed_point_similarity_inner_loop(i, j, FP_i, FP_j, labels_i, labels_j, method, seed=42):
    # Convert FP_i and FP_j to NumPy arrays if they are lists
    FP_i = np.array(FP_i) if isinstance(FP_i, list) else FP_i
    FP_j = np.array(FP_j) if isinstance(FP_j, list) else FP_j

    _, score_1 = ICP_registration(
        points_source=FP_j, labels_source=labels_j,
        points_target=FP_i, labels_target=labels_i, method=method, seed=seed
    )

    _, score_2 = ICP_registration(
        points_source=FP_i, labels_source=labels_i,
        points_target=FP_j, labels_target=labels_j, method=method, seed=seed
    )
    score = (score_1 + score_2) / 2
    return (i, j), score

def register_fixed_points(points_source, points_target, labels_source, labels_target, max_iter, tol,
                          method='procrustes', seed=42):
    c = 0
    change = np.inf
    mse_prev = np.inf
    mse = np.inf
    Q = ortho_group.rvs(dim=points_target.shape[1], random_state=seed)
    Q = Q[:points_source.shape[1], :points_target.shape[1]]
    while c < max_iter and change > tol:
        # For each source point, find a matching target point
        D = get_distance_matrix(points_source @ Q, labels_source, points_target, labels_target)
        # Clean this matrix of all the rows not being assigned
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
        c += 1
    return Q, mse

def ICP_registration(points_source, labels_source, points_target, labels_target,
                     n_tries=60, max_iter=1000, tol=1e-10, method="procrustes", seed=42):
    results = []
    for t in range(n_tries):
        results.append(register_fixed_points(points_source, points_target, labels_source, labels_target,
                                             max_iter, tol, method=method, seed=seed + t))
    Qs = [el[0] for el in results]
    mses = [el[1] for el in results]
    best_mse = np.min(mses)
    best_Q = Qs[np.argmin(mses)]
    return best_Q, best_mse

def get_distance_matrix(points_source, labels_source, points_target, labels_target):
    types = np.unique(np.concatenate([labels_source, labels_target]))
    distance = np.inf * np.ones((len(points_source), len(points_target)))
    for type in types:
        inds_source = np.where(labels_source == type)[0]
        inds_target = np.where(labels_target == type)[0]
        if len(inds_source) > 0 and len(inds_target) > 0:
            points_of_type_source = np.repeat(points_source[inds_source, np.newaxis, :], repeats=len(inds_target), axis=1)
            points_of_type_target = np.repeat(points_target[np.newaxis, inds_target, :], repeats=len(inds_source), axis=0)
            distance[np.ix_(inds_source, inds_target)] = np.linalg.norm(points_of_type_source - points_of_type_target, axis=2)
    return distance


def get_fp_similarity(fp_dict_combined, method='procrustes', seed=42):
    fp_list = [np.array(fp) for fp in fp_dict_combined["fp_list"]]  # Convert once
    labels_list = fp_dict_combined["labels_list"]
    n = len(fp_list)
    Mat = np.zeros((n, n))

    futures = []
    for i, j in tqdm(list(itertools.combinations(range(n), 2))):
        futures.append(
            computing_fixed_point_similarity_inner_loop.remote(i, j,
                                                               fp_list[i], fp_list[j],
                                                               labels_list[i],
                                                               labels_list[j],
                                                               method=method,
                                                               seed=seed + i*j + j)
        )

    print("Now fetching the results from cores")
    results = []
    remaining_futures = futures.copy()
    with tqdm(total=len(futures), desc="Fetching results", unit="tasks") as pbar:
        while remaining_futures:
            done, remaining_futures = ray.wait(remaining_futures, num_returns=min(2000, len(remaining_futures)))
            results.extend(ray.get(done))
            pbar.update(len(done))

    print("Fetched the results from cores")
    for (i, j), score in results:
        Mat[i, j] = Mat[j, i] = score
    return Mat

def get_fp_data_dict(path_to_folder, net_types, control_type_searched, dataSegment, n_nets, n_PCs):
    folder = os.listdir(path_to_folder)
    counter_dict = {}
    data_dict = {}
    for net_type in net_types:
        data_dict[net_type] = {}
        for constrained in [True, False]:
            data_dict[net_type][f"constrained={constrained}"] = {}
            for control in [False, True]:
                data_dict[net_type][f"constrained={constrained}"][f"control={control}"] = {}
                data_dict[net_type][f"constrained={constrained}"][f"control={control}"]["fp_list"] = []
                data_dict[net_type][f"constrained={constrained}"][f"control={control}"]["labels_list"] = []
                data_dict[net_type][f"constrained={constrained}"][f"control={control}"]["fp_dict_list"] = []
                counter_dict[f"{net_type}_constrained={constrained}_control={control}"] = 0

    for file in folder:
        if file == ".DS_Store" or file == "fp old":
            continue
        if ("control=False" in file) or (("control=True" in file) and (f"controlType={control_type_searched}" in file)):
            net_type = file.split("_")[0]
            segment = file.split("_")[-2]
            if segment == dataSegment:
                constrained = eval(file.split("constrained=")[1].split("_")[0])
                control = eval(file.split("control=")[1].split("_")[0])
                control_type = None if not control else file.split("controlType=")[1].split("_")[0]
                path = os.path.join(path_to_folder, file)
                opened_file = open(path, "rb+")
                fixed_points_dict = pickle.load(opened_file)
                fixed_points = fixed_points_dict["fps"]
                labels = fixed_points_dict["labels"]

                if counter_dict[f"{net_type}_constrained={constrained}_control={control}"] > n_nets - 1: # enough networks is gathered
                    pass
                else:
                    if type(fixed_points) == list: # it sometimes happens that in a control network there are no fixed points
                        F = np.zeros((1, n_PCs))
                    else:
                        F = np.zeros((fixed_points.shape[0], n_PCs))
                        n_features = min(n_PCs, fixed_points.shape[0])
                        pca = PCA(n_components=n_features)
                        F[:, :n_features] = pca.fit_transform(fixed_points)
                        F = (F - np.mean(F, axis=0))
                        R = np.sqrt(np.mean(np.sum(F ** 2, axis=1)))
                        F = F / R # remove the scale
                        if F.shape[1] != n_PCs:
                            raise ValueError("Something is wrong here!")
                        # print(f"{net_type}_constrained={constrained}_control={control}; Explained variance by {n_PCs} PCs: {np.sum(pca.explained_variance_ratio_)} R:{R}")
                    d = {}
                    input_IDs = np.unique(np.array([int(l.split("_")[-1]) for l in labels]))
                    for i in input_IDs:
                        for point_type in ["sfp", "ufp"]:
                            if f"{point_type}_{i}" in np.unique(labels):
                                inds = np.where(np.array(labels) == f"{point_type}_{i}")
                                d[f"{point_type}_{i}"] = F[inds]
                            else:
                                d[f"{point_type}_{i}"] = np.array([])

                    data_dict[net_type][f"constrained={constrained}"][f"control={control}"]["fp_list"].append(np.copy(F))
                    data_dict[net_type][f"constrained={constrained}"][f"control={control}"]["labels_list"].append(np.copy(labels))
                    data_dict[net_type][f"constrained={constrained}"][f"control={control}"]["fp_dict_list"].append(copy(d))
                    counter_dict[f"{net_type}_constrained={constrained}_control={control}"] += 1
    return data_dict

save = True
show = True
@hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'base')
def fixed_point_analysis(cfg):
    os.environ["NUMEXPR_MAX_THREADS"] = "50"
    n_nets = cfg.n_nets
    dataSegment = cfg.dataSegment
    taskname = cfg.task.taskname
    path_to_folder = os.path.join(cfg.paths.fixed_points_data_folder, taskname)
    aux_datasets_folder = os.path.join(f"{cfg.paths.auxilliary_datasets_path}", taskname)
    n_PCs = cfg.task.dynamical_topology_analysis.n_PCs
    control_type_searched = cfg.control_type
    net_types = ["relu", "sigmoid", "tanh"]
    data_dict = get_fp_data_dict(path_to_folder, net_types, control_type_searched, dataSegment, n_nets, n_PCs)
    seed = cfg.seed
    if not cfg.paths.local:
        ray.init(ignore_reinit_error=True, address="auto")
    else:
        ray.init(ignore_reinit_error=True)
    print(ray.available_resources())

    file_path = os.path.join(aux_datasets_folder, f"FPs_{dataSegment}{n_nets}_{control_type_searched}.pkl")
    if not os.path.exists(file_path):
        fp_dict_combined = {}
        fp_dict_combined["fp_list"] = []
        fp_dict_combined["labels_list"] = []
        legends = []
        inds_list = []
        cnt = 0

        # combine the fixed points together in one dataset
        for net_type in net_types:
            for constrained in [True, False]:
                for control in [False, True]:
                    fp_dict = data_dict[net_type][f"constrained={constrained}"][f"control={control}"]
                    n_nets = len(fp_dict["fp_list"])
                    # n_nets_taken = min([n_nets_MDS, n_nets])
                    n_nets_taken = n_nets
                    fp_dict_combined["fp_list"].extend([fp_dict["fp_list"][i] for i in range(n_nets_taken)])
                    fp_dict_combined["labels_list"].extend([fp_dict["labels_list"][i] for i in range(n_nets_taken)])
                    inds_list.append(cnt + np.arange(n_nets_taken))
                    cnt += n_nets_taken
                    legends.append(f"{net_type}_constrained={constrained}_control={control}")
        fp_dict_combined["inds_list"] = inds_list
        fp_dict_combined["legends"] = legends
        pickle.dump(fp_dict_combined, open(file_path, "wb+"))
    fp_dict_combined = pickle.load(open(file_path, "rb+"))
    file_path = os.path.join(aux_datasets_folder, f"FPs_similarity_matrix_{dataSegment}{n_nets}_{control_type_searched}.pkl")
    if not os.path.exists(file_path):
        Mat = get_fp_similarity(fp_dict_combined, method='procrustes', seed=seed)
        pickle.dump(Mat, open(file_path, "wb+"))
    ray.shutdown()

if __name__ == '__main__':
    fixed_point_analysis()
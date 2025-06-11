import os
import itertools
from scipy.linalg import orthogonal_procrustes
from scipy.stats import ortho_group
from sklearn.decomposition import PCA
import numpy as np
import hydra
from omegaconf import OmegaConf
import ray
from tqdm.auto import tqdm
import pickle
from trainRNNbrain.training.training_utils import prepare_task_arguments
from activation_matters.utils.trajectories_utils import get_trajectories


# Ray remote function for parallel computation
@ray.remote(num_cpus=1)  # Limit each task to 1 CPU
def computing_selectivities_similarity_inner_loop(i, j, Ti, Tj, seed=42):
    # Compute similarity (assuming ICP_registration is symmetric)
    _, score_1 = ICP_registration(points_source=Ti, points_target=Tj, max_iter=1000, tol=1e-10, seed=seed)
    _, score_2 = ICP_registration(points_source=Tj, points_target=Ti, max_iter=1000, tol=1e-10, seed=seed)
    score = (score_1 + score_2) / 2
    return (i, j), score

@ray.remote(num_cpus=1)
def extract_feature(trajectory, n_PCs):
    X = trajectory.reshape(trajectory.shape[0], -1)
    pca = PCA(n_components=n_PCs)
    selectivities = pca.fit_transform(X)
    # Remove the scale
    mean = np.mean(selectivities, axis=0)
    selectivities_normalized = (selectivities - mean) / np.sqrt(np.sum(np.var(selectivities - mean, axis=0)))
    return selectivities_normalized

def register_point_cloud(points_source, points_target, max_iter, tol, seed):
    c = 0
    change = np.inf
    mse_prev = np.inf
    mse = np.inf
    Q = ortho_group.rvs(dim=points_target.shape[1], random_state=seed)
    Q = Q[:points_source.shape[1], :points_target.shape[1]]
    while c < max_iter and change > tol:
        # For each source point, find a matching target point
        D = get_distance_matrix(points_source @ Q, points_target)
        points_target_matched = points_target[np.argmin(D, axis=1), :]
        Q, _ = orthogonal_procrustes(points_source, points_target_matched)
        mse = np.sqrt(np.sum((points_source @ Q - points_target_matched) ** 2))
        change = (mse_prev - mse)
        mse_prev = mse
        c += 1
    return Q, mse

def ICP_registration(points_source, points_target, n_tries=60, max_iter=1000, tol=1e-10, seed=42):
    results = []
    for t in range(n_tries):
        results.append(register_point_cloud(points_source, points_target, max_iter, tol, seed=seed + t))
    Qs = [el[0] for el in results]
    mses = [el[1] for el in results]
    best_mse = np.min(mses)
    best_Q = Qs[np.argmin(mses)]
    return best_Q, best_mse

def get_distance_matrix(points_source, points_target):
    source_sq = np.sum(points_source ** 2, axis=1, keepdims=True)  # (N, 1)
    target_sq = np.sum(points_target ** 2, axis=1, keepdims=True).T  # (1, M)
    cross = np.dot(points_source, points_target.T)  # (N, M)
    dist_sq = source_sq + target_sq - 2 * cross
    dist_sq = np.maximum(dist_sq, 0)
    return np.sqrt(dist_sq)

def get_selectivities_similarity(selectivities_list, seed=42):
    n = len(selectivities_list)
    Mat = np.zeros((n, n))

    # Submit all tasks to Ray
    futures = []
    for i, j in tqdm(list(itertools.combinations(range(n), 2))):
        futures.append(computing_selectivities_similarity_inner_loop.remote(i, j,
                                                                            selectivities_list[i], selectivities_list[j],
                                                                            seed=seed + i*j + j))

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

@hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'base')
def selectivity_analysis(cfg):
    os.environ["NUMEXPR_MAX_THREADS"] = "50"
    n_nets = cfg.n_nets
    dataSegment = cfg.dataSegment
    taskname = cfg.task.taskname
    dataset_path = os.path.join(f"{cfg.paths.RNN_dataset_path}", f"{taskname}_{dataSegment}{n_nets}.pkl")
    aux_datasets_folder = os.path.join(f"{cfg.paths.auxilliary_datasets_path}", taskname)
    dataset = pickle.load(open(dataset_path, "rb"))
    n_PCs = cfg.task.selectivities_analysis_params.n_PCs
    control_type = cfg.control_type  # shuffled or untrained
    seed = cfg.seed

    file_path = os.path.join(aux_datasets_folder, f"selectivities_{dataSegment}{n_nets}_{control_type}.pkl")
    if not os.path.exists(file_path):
        # Printing RNN scores for top n_nets networks
        for activation in dataset.keys():
            for constraint in dataset[activation].keys():
                scores = dataset[activation][constraint]["R2_score"]
                m = np.mean(scores)
                s = np.std(scores)
                print(f"{activation}, {constraint} R2 score = {np.round(m, 5)} +- {np.round(s, 5)}")

        # Defining the task
        task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.task.dt)
        task = hydra.utils.instantiate(task_conf)
        if taskname == "CDDM":
            task.coherences = np.array(list(cfg.task.trajectory_analysis_params.coherences))
        if hasattr(task, 'random_window'):
            task.random_window = 0  # Eliminating any source of randomness while analysing the trajectories
        task.seed = seed  # For consistency

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

                    # Assumes that all the RNNs of the same type have the same activation slope
                    activation_slope = dataset[activation_name][f"Dale={constrained}"]["activation_slope"].tolist()[0]
                    trajectories = get_trajectories(dataset=dataset[activation_name][f"Dale={constrained}"],
                                                    task=task,
                                                    activation_name=activation_name,
                                                    activation_slope=activation_slope,
                                                    get_batch_args={},
                                                    dt=1, tau=10,
                                                    shuffled=shuffled,
                                                    random=random,
                                                    constrained=constrained,
                                                    seed=seed)
                    RNN_features.append(trajectories)
                    inds_list.append(cnt + np.arange(len(trajectories)))
                    cnt += len(trajectories)

        RNN_features = list(itertools.chain.from_iterable(RNN_features))

        # Launch tasks in parallel
        if not cfg.paths.local:
            ray.init(ignore_reinit_error=True, address="auto")
        else:
            ray.init(ignore_reinit_error=True)
        print(ray.available_resources())

        results = [extract_feature.remote(feature, n_PCs) for feature in RNN_features]

        RNN_features_processed = []
        for res in tqdm(results, desc="Extracting features"):
            RNN_features_processed.append(ray.get(res))

        ray.shutdown()

        data_dict = {
            f"RNN_selectivities": RNN_features,
            f"RNN_selectivities_processed": RNN_features_processed,
            "legends": legends,
            "inds_list": inds_list
        }
        with open(file_path, "wb") as f:
            pickle.dump(data_dict, f)

    data_dict = pickle.load(open(file_path, "rb"))
    file_path = os.path.join(aux_datasets_folder, f"selectivities_similarity_matrix_{dataSegment}{n_nets}_{control_type}.pkl")

    # Calculate the similarity between the selectivities
    if not os.path.exists(file_path):
        print(f"Calculating the similarity between the selectivities")
        key = "RNN_selectivities_processed"
        if "RNN_tunings_list" in data_dict.keys():
            key = "RNN_tunings_list"
        RNN_features_processed = data_dict[key]
        Mat = get_selectivities_similarity(RNN_features_processed, seed=seed)
        with open(file_path, "wb") as f:
            pickle.dump(Mat, f)

if __name__ == '__main__':
    selectivity_analysis()
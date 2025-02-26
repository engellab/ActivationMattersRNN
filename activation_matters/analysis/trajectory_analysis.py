import numpy as np
np.set_printoptions(suppress=True)
from trainRNNbrain.training.training_utils import *
import hydra
from omegaconf import OmegaConf
from activation_matters.utils.trajectories_utils import *
from itertools import chain
import ray
# OmegaConf.register_new_resolver("eval", eval)

@ray.remote
def computing_trajectory_similarity_inner_loop(i, j, Fi, Fj):
    M1, residuals, rank, s = np.linalg.lstsq(Fi, Fj, rcond=None)
    score1 = np.sqrt(np.sum((Fi @ M1 - Fj) ** 2))
    M2, residuals, rank, s = np.linalg.lstsq(Fi, Fj, rcond=None)
    score2 = np.sqrt(np.sum((Fi @ M2 - Fj) ** 2))
    return (i, j), (score1 + score2) / 2

@ray.remote
def process_trajectory(trajectory, n_dim):
    projected_trajectory = project_trajectories(trajectory, n_dim=n_dim)
    # Divide by overall variance
    R = np.sqrt(np.sum(np.var(projected_trajectory.reshape(projected_trajectory.shape[0], -1), axis=1)))
    projected_trajectory_normalized = projected_trajectory / R
    return projected_trajectory_normalized

def get_trajectory_similarity(feature_list):
    n_dim = feature_list[0].shape[0]
    Mat = np.zeros((len(feature_list), len(feature_list)))

    elements = []
    for i in tqdm(range(len(feature_list))):
        row = []
        for j in range(i + 1, len(feature_list)):
            Fi = feature_list[i].reshape(n_dim, -1).T
            Fj = feature_list[j].reshape(n_dim, -1).T
            row.append(computing_trajectory_similarity_inner_loop.remote(i, j, Fi, Fj))
        elements.extend(ray.get(row))

    for element in elements:
        inds = element[0]
        score = element[1]
        Mat[inds] = Mat[inds[::-1]] = score
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

show = False
save = True
@hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'base')
def trajectory_analysis(cfg):
    n_nets = cfg.n_nets
    dataSegment = cfg.dataSegment
    taskname = cfg.task.taskname
    dataset_path = os.path.join(f"{cfg.paths.RNN_dataset_path}", f"{taskname}_{dataSegment}{n_nets}.pkl")
    aux_datasets_folder = os.path.join(f"{cfg.paths.auxilliary_datasets_path}", taskname)
    dataset = pickle.load(open(dataset_path, "rb"))
    n_PCs = cfg.task.trajectory_analysis_params.n_PCs

    ray.init(ignore_reinit_error=True)

    file_path = os.path.join(aux_datasets_folder, f"trajectories_data_{dataSegment}{n_nets}.pkl")
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
        shuffle_list = [False, True]
        RNN_trajectories = []
        legends = []
        inds_list = []
        cnt = 0

        for activation_name in activations_list:
            for constrained in constrained_list:
                for shuffle in shuffle_list:
                    print(f"{activation_name};Dale={constrained};shuffled={shuffle}", len(dataset[activation_name][f"Dale={constrained}"]))
                    legends.append(f"{activation_name} Dale={constrained} shuffle={shuffle}")

                    get_traj = get_trajectories_shuffled_connectivity if shuffle == True else get_trajectories
                    # assumes that all the RNNs of the same type have the same activation slope
                    activation_slope = dataset[activation_name][f"Dale={constrained}"]["activation_slope"].tolist()[0]
                    trajectories = get_traj(dataset=dataset[activation_name][f"Dale={constrained}"],
                                   task=task,
                                   activation_name=activation_name,
                                   activation_slope=activation_slope,
                                   get_batch_args={})
                    RNN_trajectories.append(trajectories)
                    inds_list.append(cnt + np.arange(len(trajectories)))
                    cnt += len(trajectories)

        RNN_trajectories = list(chain.from_iterable(RNN_trajectories))
        RNN_trajectories_projected = []

        # Launch tasks in parallel
        results = [process_trajectory.remote(trajectory, n_PCs) for trajectory in RNN_trajectories]
        for res in tqdm(results):
            RNN_trajectories_projected.append(ray.get(res))

        data_dict = {}
        data_dict["RNN_trajectories"] = RNN_trajectories
        data_dict["RNN_trajectories_projected"] = RNN_trajectories_projected
        data_dict["legends"] = legends
        data_dict["inds_list"] = inds_list
        pickle.dump(data_dict, open(file_path, "wb+"))

    data_dict = pickle.load(open(file_path, "rb+"))
    file_path = os.path.join(aux_datasets_folder, f"trajectories_similarity_matrix_{dataSegment}{n_nets}.pkl")

    # GET THE SIMILARITY BETWEEN THE TRAJECTORIES
    if not os.path.exists(file_path):
        print("Calculating the similarity between the trajectories")
        RNN_trajectories_projected = data_dict["RNN_trajectories_projected"]
        Mat = get_trajectory_similarity(RNN_trajectories_projected)
        pickle.dump(Mat, open(file_path, "wb+"))

    ray.shutdown()

if __name__ == '__main__':
    trajectory_analysis()
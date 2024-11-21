import numpy as np
np.set_printoptions(suppress=True)
from trainRNNbrain.training.training_utils import *
import hydra
from omegaconf import OmegaConf
from activation_matters.utils.trajectories_utils import *
from itertools import chain
OmegaConf.register_new_resolver("eval", eval)

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


taskname = "CDDM"
show = False
save = True
# @hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'base')
@hydra.main(version_base="1.3", config_path=f"../../configs/task", config_name=f'{taskname}')
def analysis_of_trajectories(cfg):
    taskname = cfg.task.taskname
    dataset_path = os.path.join(f"{cfg.task.paths.dataset_path}", f"{taskname}_top30.pkl")
    aux_datasets_folder = f"{cfg.task.paths.auxilliary_datasets_path}"
    dataset = pickle.load(open(dataset_path, "rb"))
    n_PCs = cfg.task.trajectory_analysis_params.n_PCs
    #printing RNN scores for top 30 networks
    for activation in dataset.keys():
        for constraint in dataset[activation].keys():
            scores = dataset[activation][constraint]["RNN_score"]
            m = np.mean(scores)
            s = np.std(scores)
            print(f"{activation}, {constraint} score = {np.round(m, 5)} +- {np.round(s, 5)}")

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
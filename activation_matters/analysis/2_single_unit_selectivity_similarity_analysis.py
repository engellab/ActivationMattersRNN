from scipy.stats import ortho_group
from trainRNNbrain.training.training_utils import *
from activation_matters.utils.trajectories_utils import *
import hydra
from omegaconf import OmegaConf
np.set_printoptions(suppress=True)
OmegaConf.register_new_resolver("eval", eval)

def ICP_registration(points_source, points_target, n_tries=31, max_iter=1000, tol=1e-10):
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
            D = get_distance_matrix(points_source @ Q, points_target)
            points_target_matched = points_target[np.argmin(D, axis=1), :]
            Q, _ = orthogonal_procrustes(points_source, points_target_matched)
            mse = np.sqrt(np.sum((points_source @ Q - points_target_matched) ** 2))
            change = (mse_prev - mse)
            mse_prev = mse
            i += 1
        Qs.append(np.copy(Q))
        mses.append(mse)
    best_mse = np.min(mses)
    best_Q = Qs[np.argmin(mses)]
    return best_Q, best_mse

def get_distance_matrix(points_source, points_target):
    points_source_ = np.repeat(points_source[:, np.newaxis, :], repeats=points_target.shape[0], axis=1)
    points_target_ = np.repeat(points_target[np.newaxis, :, :], repeats=points_source.shape[0], axis=0)
    distance = np.linalg.norm(points_source_ - points_target_, axis = 2)
    return distance

def get_neurrep_similarity(neurrep_list):
    Mat = np.zeros((len(neurrep_list), len(neurrep_list)))
    for i in tqdm(range(len(neurrep_list))):
        for j in (range(i + 1, len(neurrep_list))):
            _, score_1 = ICP_registration(points_source=neurrep_list[i],
                                        points_target=neurrep_list[j],
                                        max_iter=1000, tol=1e-10)
            _, score_2 = ICP_registration(points_source=neurrep_list[j],
                                        points_target=neurrep_list[i],
                                        max_iter=1000, tol=1e-10)
            Mat[i, j] = Mat[j, i] = (score_1 + score_2) / 2
    return Mat

def get_neurreps(trajectory, n_dim = 5):
    pca = PCA(n_components=n_dim)
    trajectories_flattened = trajectory.reshape(trajectory.shape[0], -1)
    N = trajectory.shape[0]
    T = trajectory.shape[1]
    K = trajectory.shape[2]
    res = pca.fit_transform(trajectories_flattened.T).T.reshape(n_dim, T, K)
    print(f"Explained variance by {n_dim} PCs: {np.sum(pca.explained_variance_ratio_)}")
    return res


taskname = "CDDM"
show = True
save = False

# @hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'base')
@hydra.main(version_base="1.3", config_path=f"../../configs/task", config_name=f'{taskname}')
def analysis_of_neural_representations(cfg):
    taskname = cfg.task.taskname
    dataset_path = os.path.join(f"{cfg.task.paths.RNN_dataset_path}", f"{taskname}_top30.pkl")
    aux_datasets_folder = f"{cfg.task.paths.auxilliary_datasets_path}"
    dataset = pickle.load(open(dataset_path, "rb"))

    # defining the task
    task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.task.dt)
    task = hydra.utils.instantiate(task_conf)
    if hasattr(task, 'random_window'):
        task.random_window = 0 # eliminating any source of randomness while analysing neural representations
    task.seed = 0 # for consistency

    n_PCs = cfg.task.neurreps_analysis_params.n_PCs

    activations_list = ["relu", "sigmoid", "tanh"]
    constrained_list = [True, False]
    shuffle_list = [False, True]
    legends = []
    inds_list = []
    RNN_neurreps_list = []
    cnt = 0

    for activation_name in activations_list:
        for constrained in constrained_list:
            for shuffle in shuffle_list:
                print(f"{activation_name};Dale={constrained};shuffled={shuffle}", len(dataset[activation_name][f"Dale={constrained}"]))

                legends.append(f"{activation_name} Dale={constrained} shuffle={shuffle}")

                # assumes that all the RNNs of the same type have the same activation slope
                activation_slope = dataset[activation_name][f"Dale={constrained}"]["activation_slope"].tolist()[0]

                get_traj = get_trajectories_shuffled_connectivity if shuffle == True else get_trajectories
                RNN_trajectories = get_traj(dataset=dataset[activation_name][f"Dale={constrained}"],
                                       task=task,
                                       activation_name=activation_name,
                                       activation_slope=activation_slope,
                                       get_batch_args={})

                # get global PCs
                X = np.vstack([trajectory.reshape(-1, trajectory.shape[1] * trajectory.shape[-1]) for trajectory in RNN_trajectories])

                pca = PCA(n_components=n_PCs)
                pca.fit(X)
                PCs = pca.components_
                print(f"Variance explained by first {n_PCs} PCs: {np.sum(pca.explained_variance_ratio_)}")

                # # get activity threshold
                # m = np.mean(np.abs(X), axis=1)
                # thr = np.quantile(m, q)

                for i in tqdm(range(len(RNN_trajectories))):
                    # inds = np.where(np.mean(np.mean(np.abs(RNN_trajectories[i]), axis=1), axis=1) >= thr)[0]
                    # trajectory = RNN_trajectories[i][inds, ...]
                    trajectory = RNN_trajectories[i]
                    neurrep = trajectory.reshape(trajectory.shape[0], trajectory.shape[1] * trajectory.shape[2]) @ PCs.T

                    # remove the scale
                    mean = np.repeat(np.mean(neurrep, axis = 0).reshape(1, -1), axis = 0, repeats=neurrep.shape[0])
                    neurrep_normalized = (neurrep - mean) / np.sqrt(np.sum(np.var((neurrep - mean), axis=0)))

                    RNN_neurreps_list.append(np.copy(neurrep_normalized))

                inds_list.append(cnt + np.arange(len(RNN_trajectories)))
                cnt += len(RNN_trajectories)

    data_dict = {}
    data_dict["RNN_neurreps_list"] = RNN_neurreps_list
    data_dict["inds_list"] = inds_list
    data_dict["legends"] = legends
    pickle.dump(data_dict, open(os.path.join(aux_datasets_folder, "neurrep_similarity.pkl"), "wb+"))

    # GET THE SIMILARITY BETWEEN THE TRAJECTORIES
    print("Calculating the similarity between the neural representations")
    Mat = get_neurrep_similarity(RNN_neurreps_list)
    pickle.dump(Mat, open(os.path.join(aux_datasets_folder, "neurreps_similarity_matrix.pkl"), "wb"))


if __name__ == '__main__':
    analysis_of_neural_representations()
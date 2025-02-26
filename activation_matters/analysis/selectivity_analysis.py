from scipy.stats import ortho_group
from trainRNNbrain.training.training_utils import *
from activation_matters.utils.trajectories_utils import *
import hydra
from omegaconf import OmegaConf
import ray
np.set_printoptions(suppress=True)
# OmegaConf.register_new_resolver("eval", eval)


def register_point_cloud(points_source, points_target, max_iter, tol):
    c = 0
    change = np.inf
    mse_prev = np.inf
    mse = np.inf
    Q = ortho_group.rvs(dim=points_target.shape[1])
    Q = Q[:points_source.shape[1], :points_target.shape[1]]
    while c < max_iter and change > tol:
        # for each source point, find a matching target point
        D = get_distance_matrix(points_source @ Q, points_target)
        points_target_matched = points_target[np.argmin(D, axis=1), :]
        Q, _ = orthogonal_procrustes(points_source, points_target_matched)
        mse = np.sqrt(np.sum((points_source @ Q - points_target_matched) ** 2))
        change = (mse_prev - mse)
        mse_prev = mse
        c += 1
    return Q, mse


@ray.remote
def computing_tuning_similarity_inner_loop(i, j, Ti, Tj):
    _, score1 = ICP_registration(points_source=Ti,
                                  points_target=Tj,
                                  max_iter=1000, tol=1e-10)
    _, score2 = ICP_registration(points_source=Tj,
                                  points_target=Ti,
                                  max_iter=1000, tol=1e-10)
    return (i, j), (score1 + score2) / 2

@ray.remote
def process_tunings(trajectory, PCs):
    tuning = trajectory.reshape(trajectory.shape[0], trajectory.shape[1] * trajectory.shape[2]) @ PCs.T
    # remove the scale
    mean = np.repeat(np.mean(tuning, axis=0).reshape(1, -1), axis=0, repeats=tuning.shape[0])
    tuning_normalized = (tuning - mean) / np.sqrt(np.sum(np.var((tuning - mean), axis=0)))
    return tuning_normalized


def ICP_registration(points_source, points_target, n_tries=60, max_iter=1000, tol=1e-10):
    results = []
    for t in range(n_tries):
        results.append(register_point_cloud(points_source, points_target, max_iter, tol))
    Qs = [el[0] for el in results]
    mses = [el[1] for el in results]
    best_mse = np.min(mses)
    best_Q = Qs[np.argmin(mses)]
    return best_Q, best_mse

def get_distance_matrix(points_source, points_target):
    points_source_ = np.repeat(points_source[:, np.newaxis, :], repeats=points_target.shape[0], axis=1)
    points_target_ = np.repeat(points_target[np.newaxis, :, :], repeats=points_source.shape[0], axis=0)
    distance = np.linalg.norm(points_source_ - points_target_, axis = 2)
    return distance


def get_tuning_similarity(tuning_list):
    Mat = np.zeros((len(tuning_list), len(tuning_list)))

    elements = []
    for i in tqdm(range(len(tuning_list))):
        row = []
        for j in range(i + 1, len(tuning_list)):
            row.append(computing_tuning_similarity_inner_loop.remote(i, j, tuning_list[i], tuning_list[j]))
        elements.extend(ray.get(row))

    for element in elements:
        inds = element[0]
        score = element[1]
        Mat[inds] = Mat[inds[::-1]] = score
    return Mat

@hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'base')
def selectivity_analysis(cfg):
    n_nets = cfg.n_nets
    dataSegment = cfg.dataSegment
    taskname = cfg.task.taskname
    dataset_path = os.path.join(f"{cfg.paths.RNN_dataset_path}", f"{taskname}_{dataSegment}{n_nets}.pkl")
    aux_datasets_folder = os.path.join(f"{cfg.paths.auxilliary_datasets_path}", taskname)
    dataset = pickle.load(open(dataset_path, "rb"))
    ray.init()

    file_path = os.path.join(aux_datasets_folder, f"tunings_{dataSegment}{n_nets}.pkl")
    if not os.path.exists(file_path):
        # defining the task
        task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.task.dt)
        task = hydra.utils.instantiate(task_conf)
        if hasattr(task, 'random_window'):
            task.random_window = 0 # eliminating any source of randomness while analysing neural representations
        task.seed = 0 # for consistency

        n_PCs = cfg.task.tunings_analysis_params.n_PCs

        activations_list = ["relu", "sigmoid", "tanh"]
        constrained_list = [True, False]
        shuffle_list = [False, True]
        legends = []
        inds_list = []
        RNN_tunings_list = []
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

                    # Launch tasks in parallel
                    results = [process_tunings.remote(trajectory, PCs) for trajectory in RNN_trajectories]
                    for res in tqdm(results):
                        RNN_tunings_list.append(ray.get(res))

                    inds_list.append(cnt + np.arange(len(RNN_trajectories)))
                    cnt += len(RNN_trajectories)

        data_dict = {}
        data_dict["RNN_tunings_list"] = RNN_tunings_list
        data_dict["inds_list"] = inds_list
        data_dict["legends"] = legends
        pickle.dump(data_dict, open(file_path, "wb+"))

    data_dict = pickle.load(open(file_path, "rb+"))
    file_path = os.path.join(aux_datasets_folder, f"tunings_similarity_matrix_{dataSegment}{n_nets}.pkl")
    if not os.path.exists(file_path):
        # GET THE SIMILARITY BETWEEN THE TRAJECTORIES
        print("Calculating the similarity between the neural tuning configurations")
        RNN_tunings_list = data_dict["RNN_tunings_list"]
        Mat = get_tuning_similarity(RNN_tunings_list)
        pickle.dump(Mat, open(file_path, "wb+"))
    ray.shutdown()


if __name__ == '__main__':
    selectivity_analysis()
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.stats import ortho_group
from src.plots.ploting_utils import plot_similarity_matrix, plot_embedding, plot_representations
from sklearn.manifold import MDS
from rnn_coach.training.training_utils import *
from src.utils.trajectories_utils import *
import hydra
from omegaconf import OmegaConf
from style.style_setup import set_up_plotting_styles
from itertools import chain
from copy import deepcopy
np.set_printoptions(suppress=True)

OmegaConf.register_new_resolver("eval", eval)
taskname = "MemoryNumber"
show = False
save = True
n_nets = 30
n_PCs = 5

def mean_distance_to_centroid(points):
    # Calculate the centroid
    centroid = np.mean(points, axis=0)
    # Calculate the mean distance to the centroid
    distances = np.linalg.norm(points - centroid, axis=1)
    return np.mean(distances)

def standard_deviation_distance(points):
    # Calculate the centroid
    centroid = np.mean(points, axis=0)
    # Calculate the distances to the centroid
    distances = np.linalg.norm(points - centroid, axis=1)
    return np.std(distances)

def radius_of_gyration(points):
    # Calculate the centroid
    centroid = np.mean(points, axis=0)
    # Calculate the squared distances to the centroid
    squared_distances = np.sum((points - centroid)**2, axis=1)
    return np.sqrt(np.mean(squared_distances))

def meadian_2pointdist(points):
    N = points.shape[0]
    points1 = np.repeat(points[np.newaxis,:, :], N, axis=0)
    points2 = np.repeat(points[:, np.newaxis, :], N, axis=1)
    D = np.sqrt(np.sum(((points1 - points2)**2), axis = -1))
    return np.median(D)

def median_dist_to_closest(points):
    N = points.shape[0]
    points1 = np.repeat(points[np.newaxis,:, :], N, axis=0)
    points2 = np.repeat(points[:, np.newaxis, :], N, axis=1)
    D = np.sqrt(np.sum(((points1 - points2)**2), axis = -1))
    np.fill_diagonal(D, np.inf)
    min_dists = np.min(D, axis=1)
    return np.median(min_dists)

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
            if neurrep_list[i].shape[0] > neurrep_list[j].shape[0]:
                neurrep_target = neurrep_list[i]
                neurrep_source = neurrep_list[j]
            else:
                neurrep_target = neurrep_list[j]
                neurrep_source = neurrep_list[i]
            M, score = ICP_registration(points_source=neurrep_source,
                                        points_target=neurrep_target,
                                        max_iter=1000, tol=1e-10)
            Mat[i, j] = Mat[j, i] = score
    return Mat


@hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'{taskname}_conf')
def analysis_of_neural_representations(cfg):
    set_up_plotting_styles(cfg.paths.style_path)
    img_folder = cfg.paths.img_folder
    aux_datasets_folder = cfg.paths.auxilliary_datasets_path
    # defining the task
    task_conf = prepare_task_arguments(cfg_task=cfg, dt=cfg.dt)
    task = hydra.utils.instantiate(task_conf)
    if hasattr(task, 'random_window'):
        task.random_window = 0
    task.seed = 0 # for consistency
    get_batch_args = {}
    if taskname == "DMTS":
        get_batch_args = {"num_rep" : 4}

    n_steps = task_conf.n_steps
    downsample_window = cfg.feature_extraction_params.downsample_window

    DF_dict = {}
    file_str = os.path.join(cfg.paths.RNN_datasets_path, f"{taskname}.pkl")
    activations_list = ["relu", "sigmoid", "tanh"]
    constrained_list = [True, False]
    RNN_neurreps_list = []
    legends = []
    inds_list = []
    cnt = 0
    try:
        data_dict = pickle.load(open(os.path.join(aux_datasets_folder, "neurrep_similarity.pkl"), "rb+"))
    except:
        for activation_name in activations_list:
            DF_dict[activation_name] = {}
            for constrained in constrained_list:
                RNN_score = eval(f"cfg.dataset_filtering_params.{activation_name}_filters.RNN_score_filter")
                lambda_r = eval(f"cfg.dataset_filtering_params.{activation_name}_filters.lambda_r")
                activation_slope = eval(f"cfg.dataset_filtering_params.{activation_name}_filters.activation_slope")
                filters = {"activation_name": ("==", activation_name),
                           "activation_slope": ("==", activation_slope),
                           "RNN_score": ("<=", RNN_score),
                           "constrained": ("==", constrained),
                           "lambda_r": (">=", lambda_r)}
                dataset = get_dataset(file_str, filters)
                print(f"{activation_name};Dale={constrained}", len(dataset))
                DF_dict[activation_name][f"Dale={constrained}"] = deepcopy(dataset[:n_nets])
                dkeys = (f"{activation_name}", f"Dale={constrained}")
                legends.append(f"{activation_name} Dale={constrained}")

                trajectories = get_trajectories(dataset=DF_dict[dkeys[0]][dkeys[1]],
                                               task=task,
                                               activation_name=activation_name,
                                               activation_slope=activation_slope,
                                               get_batch_args=get_batch_args)

                # get neural representations given the trajectory
                pca = PCA(n_components=n_PCs)
                for k in range(n_nets):
                    trajectory = trajectories[k]
                    second_dim = trajectory.shape[1] * trajectory.shape[-1]
                    X = trajectory.reshape(-1, second_dim)
                    neural_representations = pca.fit_transform(X)
                    RNN_neurreps_list.append(neural_representations)
                    # for axes in [(0, 1, 2), (0, 2, 3), (1, 2, 3)]:
                    #     legend = f"{activation_name}; Dale={constrained}"
                    #     path = os.path.join(img_folder, f"{taskname}_neural_representations_{legend}_{axes}_{k}.pdf")
                    #     plot_representations(neural_representations,
                    #                          axes=axes[:2],
                    #                          labels=None,
                    #                          show=show,
                    #                          save=save, path=path,
                    #                          s=40, alpha=0.5, n_dim=2)
                inds_list.append(cnt + np.arange(n_nets))
                cnt += n_nets
        data_dict = {}
        data_dict["RNN_neurreps_list"] = RNN_neurreps_list
        data_dict["RNNs_inds_list"] = inds_list
        data_dict["legends"] = legends
        pickle.dump(data_dict, open(os.path.join(aux_datasets_folder, "neurrep_similarity.pkl"), "wb+"))

    RNN_neurreps_list = data_dict["RNN_neurreps_list"]
    inds_list = data_dict["RNNs_inds_list"]
    legends = data_dict["legends"]
    for l, legend in enumerate(legends):
        neurreps_list = [RNN_neurreps_list[k] for k in inds_list[l]]
        neurrep_target = neurreps_list[0]
        Qs = []
        RNN_neurreps_registered_list = []
        for j, neurrep_source in enumerate(neurreps_list):
            Q, mse = ICP_registration(neurrep_source, neurrep_target)
            Qs.append(np.copy(Q))
            RNN_neurreps_registered_list.append(neurrep_source @ Q)

        neural_representations_registered = np.vstack(RNN_neurreps_registered_list)

        # for axes in [(0, 1, 2), (0, 2, 3), (1, 2, 3)]:
        #     legend = legend
        #     path = os.path.join(img_folder, f"{taskname}_neural_representations_registered_{legend}_{axes}.pdf")
        #     plot_representations(neural_representations_registered,
        #                          axes=axes[:3],
        #                          labels=None,
        #                          show=True,
        #                          save=save, path=path,
        #                          s=10, alpha=0.5, n_dim=3)

        mean_dist = mean_distance_to_centroid(neural_representations_registered)
        std_dev = standard_deviation_distance(neural_representations_registered)
        radius_gyration = radius_of_gyration(neural_representations_registered)
        med_2p_dist = meadian_2pointdist(neural_representations_registered)
        med_closest_dist = median_dist_to_closest(neural_representations_registered)

        print("#######################################")
        print(legend)
        print("Mean Distance to Centroid:", mean_dist)
        print("Standard Deviation of Distances:", std_dev)
        print("Radius of Gyration:", radius_gyration)
        print("Median 2p Distance:", med_2p_dist)
        print("Median Closest Distance:", med_closest_dist)
        ####### plot registered neurreps:
        n_dim = 3
        for axes in [(0, 1, 2), (0, 2, 3), (1, 2, 3)]:
            path = os.path.join(img_folder, f"{taskname}_neural_representations_{legend}_{axes}_registered.pdf")
            plot_representations(neural_representations_registered,
                                 axes=axes[:n_dim],
                                 labels=None,
                                 show=show,
                                 save=save, path=path,
                                 s=3, alpha=0.5, n_dim=n_dim)


if __name__ == '__main__':
    analysis_of_neural_representations()
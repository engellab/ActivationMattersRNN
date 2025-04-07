import numpy as np
np.set_printoptions(suppress=True)
from trainRNNbrain.training.training_utils import *
from activation_matters.utils.trajectories_utils import *
import hydra
from omegaconf import OmegaConf
from activation_matters.plots.style.style_setup import set_up_plotting_styles
from sklearn.decomposition import PCA
OmegaConf.register_new_resolver("eval", eval)
taskname = "MemoryNumber"

@hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'{taskname}_conf')
def analyse_varaince(cfg):
    set_up_plotting_styles(cfg.paths.style_path)
    n_nets = cfg.trajectory_analysis_params.n_nets
    task_conf = prepare_task_arguments(cfg_task=cfg, dt=cfg.dt)
    task = hydra.utils.instantiate(task_conf)
    if taskname == "CDDM":
        task.coherences = np.array([-1, -0.5, 0, 0.5, 1.0])
    if hasattr(task, 'random_window'):
        task.random_window = 0
    task.seed = 0 # for consistency
    get_batch_args = {}
    DF_dict = {}
    file_str = os.path.join(cfg.paths.RNN_datasets_path, f"{taskname}.pkl")
    activations_list = ["relu", "sigmoid", "tanh"]
    constrained_list = [True, False]
    for activation_name in activations_list:
        DF_dict[activation_name] = {}
        trajectory_list = []
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

            dkeys = (f"{activation_name}", f"Dale={constrained}", f"shuffle={False}")

            trajectories_list = get_trajectories(dataset=dataset[:n_nets],
                                           task=task,
                                           activation_name=activation_name,
                                           activation_slope=activation_slope,
                                           get_batch_args=get_batch_args)
            trajectories_combined = np.concatenate(trajectories_list, axis=0)
            N = trajectories_combined.shape[0]
            T = trajectories_combined.shape[1]
            K = trajectories_combined.shape[2]

            mean_score = np.round(np.mean(dataset[:n_nets]["RNN_score"]), 6)

            pca = PCA(n_components=np.minimum(K, 40))
            pca.fit(trajectories_combined.reshape(N*T, K))
            var3Pcs_trajectories = np.sum(pca.explained_variance_ratio_[:3])

            pca = PCA(n_components=40)
            pca.fit(trajectories_combined.reshape(N, T*K))
            var3Pcs_neurreps = np.sum(pca.explained_variance_ratio_[:3])
            #                  f"var3Pcs_trajectories={np.round(var3Pcs_trajectories,5)}, "
            print(f"{activation_name}, costrained={constrained}, "
                  f"var3Pcs_neurreps={np.round(var3Pcs_neurreps, 5)},"
                  f"mean RNN score={mean_score}")



if __name__ == '__main__':
    analyse_varaince()
from rnn_coach.training.training_utils import *
import numpy as np
from src.utils.feautre_extraction_utils import extract_features_single_RNN
from src.utils.neurons_scoring import score_neurons
from src.utils import get_ordering

np.set_printoptions(suppress=True)
import hydra
from omegaconf import OmegaConf
from src.utils.make_RNN_datasets_utils import *
from style.style_setup import set_up_plotting_styles
import re


def get_training_mask(Ts, dt):
    mask_part_list = []
    for i in range(len(Ts)):
        tuple = Ts[i]
        t1 = int(tuple[0] / dt)
        t2 = int(tuple[1] / dt)
        mask_part_list.append(t1 + np.arange(t2 - t1))
    return np.concatenate(mask_part_list)

def evaluate_expression(expr, params):
    # Replace placeholders with actual values
    expr = re.sub(r'\${\.\.([^}]+)}', lambda m: str(params[m.group(1)]), expr)
    expr = re.sub(r'\${\.([^}]+)}', lambda m: str(params[m.group(1)]), expr)
    # Evaluate the expression if it starts with 'eval:'
    if expr.startswith('${eval:'):
        expr = expr[7:-1]
        return eval(expr)
    return eval(expr)


taskname = 'CDDM'
save = True
show = False
print(taskname)
os.environ['HYDRA_FULL_ERROR'] = '1'
OmegaConf.register_new_resolver("eval", eval)
@hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'{taskname}_conf')
def analyze_single_RNN(cfg):
    set_up_plotting_styles(cfg.paths.style_path)
    img_path = cfg.paths.img_folder
    for activation in ["relu", "sigmoid", "tanh"]:
        for constrained in [True, False]:
            # defining the task
            task_conf = prepare_task_arguments(cfg_task=cfg, dt=cfg.dt)
            task = hydra.utils.instantiate(task_conf)

            RNNs_folders = collected_relevant_folders(cfg.paths.trained_RNNs_path, taskname)
            RNNs_folders = [RNN_folder for RNN_folder in RNNs_folders if (activation in RNN_folder) and (f"constrained={constrained}" in RNN_folder)]
            scores = [float(folder.split("/")[1].split("_")[0]) for folder in RNNs_folders]
            scores, RNNs_folders = zip(*sorted(zip(scores, RNNs_folders)))
            for n_RNN in np.arange(5):
                RNN_folder = RNNs_folders[n_RNN]
                print(RNN_folder)
                RNN_folder_path = os.path.join(cfg.paths.trained_RNNs_path, RNN_folder)
                RNN_data = load_data(RNN_folder_path)
                W_out = np.array(RNN_data["W_out"])
                W_inp = np.array(RNN_data["W_inp"])
                W_rec = np.array(RNN_data["W_rec"])
                RNN_config = load_config(RNN_folder_path)
                if "model" in RNN_config.keys():
                    dt = RNN_config["model"]["dt"]
                    tau = RNN_config["model"]["tau"]
                    activation_name = RNN_config["model"]["activation_name"]
                    if "activation_slope" in RNN_config["model"].keys():
                        activation_slope = RNN_config["model"]["activation_slope"]
                    else:
                        activation_slope = 1.0
                    constrained = RNN_config["model"]["constrained"]
                    for key, value in RNN_config["task"].items():
                        if key[0] == '_':
                            continue
                        if isinstance(value, str) and '${' in value:
                            RNN_config["task"][key] = evaluate_expression(value, RNN_config["task"])
                    mask_params = [evaluate_expression(param, RNN_config["task"]) for param in
                                   RNN_config["task"]['mask_params']]
                    mask = get_training_mask(mask_params, dt=dt)
                else:
                    dt = RNN_config["dt"]
                    tau = RNN_config["tau"]
                    activation_name = RNN_config["activation"]
                    if "activation_slope" in RNN_config.keys():
                        activation_slope = RNN_config["activation_slope"]
                    else:
                        activation_slope = 1.0
                    constrained = RNN_config["constrained"]
                    mask = RNN_config["mask"]


                RNN = RNN_numpy(N=W_rec.shape[0],
                                dt=dt, tau=tau,
                                W_inp=W_inp,
                                W_rec=W_rec,
                                W_out=W_out,
                                activation_name=activation_name,
                                activation_slope=activation_slope)

                mse = lambda x, y: np.sum((x - y) ** 2)
                # Evaluate all the parameters in the dictionary
                nrn_scores = score_neurons(RNN, task, scoring_function=mse, mask=mask)

                # kneedle = KneeLocator(np.arange(len(nrn_scores)), sorted(nrn_scores), S=1,
                #                       curve="convex",
                #                       direction="increasing")
                # knee = int(kneedle.knee)
                # inds = np.where(nrn_scores >= knee)[0]
                # if len(inds) < 20:
                #     knee = len(nrn_scores) - 20 # always have at least 20 neurons
                #     inds = np.where(nrn_scores >= knee)[0]

                # thr = 0.02
                # inds = np.where(nrn_scores >= thr)[0]
                # if len(inds) < 20:
                #     knee = len(nrn_scores) - 20 # always have at least 20 neurons
                #     inds = np.where(nrn_scores >= knee)[0]
                # knee = np.where(np.array(sorted(nrn_scores)) >= thr)[0][0]


                # fig, ax = plt.subplots(1, 1, figsize = (8, 5))
                # ax.plot(sorted(nrn_scores), color='r')
                # # ax.axhline(y=thr, color='k', linestyle='--')
                # ax.axvline(x=knee, color='k', linestyle='--')
                # img_name = f"Contribution_{activation_name}_constrained={constrained}_{n_RNN}.png"
                # path = os.path.join(f'../../img/{taskname}', img_name)
                # fig.savefig(path, transparent=False)
                # if show:
                #     plt.show()
                # plt.close()

                # need to create a dataset
                processing_params = {}
                if not (cfg.feature_extraction_params.Trajectory is None):
                    processing_params["Trajectory"] = cfg.feature_extraction_params.Trajectory
                if not (cfg.feature_extraction_params.Connectivity is None):
                    processing_params["Connectivity"] = cfg.feature_extraction_params.Connectivity

                # feature_data = extract_features_single_RNN(RNN_trajectories,
                #                                            W_inp, W_rec, W_out,
                #                                            downsample_window=cfg.feature_extraction_params.downsample_window,
                #                                            processing_params=processing_params,
                #                                            connectivity_importance_factor=3,
                #                                            nrn_type_importance_factor=0.25)

                feature_data = extract_features_single_RNN(RNN, task, mask,
                                                           downsample_window=cfg.feature_extraction_params.downsample_window,
                                                           processing_params=processing_params,
                                                           connectivity_importance_factor=1,
                                                           nrn_type_importance_factor=0.25)
                # # do the clustering
                # clusterer_params = {"max_iter": cfg.representations_analysis.max_iter,
                #                     "gamma": cfg.representations_analysis.gamma,
                #                     "increase_gamma": cfg.representations_analysis.increase_gamma}
                # clusterer_bldr_fn = lambda n_clusters: CustomClusterer(n_clusters, **clusterer_params)
                # model_selector = ModelSelector(clusterer_bldr_fn,
                #                                scoring_method=cfg.representations_analysis.cluster_scoring_method,
                #                                min_clusters=cfg.representations_analysis.min_clusters,
                #                                max_clusters=cfg.representations_analysis.max_clusters,
                #                                n_repeats=cfg.representations_analysis.n_repeats,
                #                                selection_criteria=cfg.representations_analysis.selection_criteria,
                #                                parallel=True)
                # F = feature_data["Features"]
                # optimal_n_clusters = model_selector.select_n_clusters(feature_list=[F],
                #                                                       W_recs=[feature_data["W_rec"]],
                #                                                       visualize=show)
                #
                # print(f"Optimal number of clusters: {optimal_n_clusters}")
                #
                # clusterer = CustomClusterer(n_clusters=optimal_n_clusters, **clusterer_params)
                # best_score, labels_list = get_best_clustering(clusterer=clusterer,
                #                                               feature_list=[F],
                #                                               W_recs=[feature_data["W_rec"]],
                #                                               scoring_method=cfg.representations_analysis.cluster_scoring_method,
                #                                               n_repeats=cfg.representations_analysis.n_repeats)
                # labels = list(chain.from_iterable(labels_list))
                #

                inds_sorted = get_ordering(np.hstack([feature_data["W_inp"], feature_data["W_out"].T]), th=0.1)
                # sort labels along with indices
                # labels, inds_sorted = zip(*sorted(zip(labels, np.arange(len(inds)))))
                # also need to reshuffle the labels according to the inputs
                W_inp_permuted = feature_data["W_inp"][inds_sorted, :]
                W_out_permuted = feature_data["W_out"][:, inds_sorted]
                W_rec_permuted = feature_data["W_rec"][:, inds_sorted]
                W_rec_permuted = W_rec_permuted[inds_sorted, :]

                N = W_rec_permuted.shape[0]
                n_inp = W_inp_permuted.shape[1]
                n_out = W_out_permuted.shape[0]
                print(n_inp, N, n_out)
                fig, ax = plt.subplots(3, 2,
                                       gridspec_kw={'width_ratios': [N, n_inp], 'height_ratios': [n_inp, N, n_out]},
                                       figsize=(4, 4))
                # Plotting the recurrent matrix in the center
                ax[1, 0].imshow(W_rec_permuted, aspect=1, cmap='bwr', vmin=-0.5, vmax=0.5)
                ax[1, 0].set_title('Recurrent Matrix', loc='center')
                ax[1, 0].axis('off')

                # Plotting the input matrix to the right
                ax[1, 1].imshow(W_inp_permuted, aspect=1, cmap='bwr', vmin=-0.5, vmax=0.5)
                ax[1, 1].set_title('Input Matrix', loc='center')
                ax[1, 1].axis('off')

                # Plotting the input matrix to the right
                ax[0, 0].imshow(W_inp_permuted.T, aspect=1, cmap='bwr', vmin=-0.5, vmax=0.5)
                ax[0, 0].set_title('Input Matrix', loc='center')
                ax[0, 0].axis('off')

                # Plotting the output matrix underneath the recurrent matrix
                ax[2, 0].imshow(W_out_permuted, aspect=1, cmap='bwr', vmin=-0.5, vmax=0.5)
                ax[2, 0].set_title('Output Matrix', loc='center')
                ax[2, 0].axis('off')

                # Hiding the empty subplot
                ax[2, 1].axis('off')
                ax[0, 1].axis('off')

                plt.tight_layout()
                path = os.path.join(img_path, f"PermutedMatrix_{activation_name}_constrained={constrained}_RNN={n_RNN}.png")
                if save:
                    fig.savefig(path, dpi=300, transparent=True, bbox_inches='tight')
                if show:
                    plt.show()


if __name__ == '__main__':
    analyze_single_RNN()
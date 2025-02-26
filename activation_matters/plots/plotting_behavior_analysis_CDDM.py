import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from omegaconf import OmegaConf
from trainRNNbrain.training.training_utils import prepare_task_arguments
from activation_matters.utils.feautre_extraction_utils import get_dataset

np.set_printoptions(suppress=True)
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.analyzers.PerformanceAnalyzer import *

import hydra
OmegaConf.register_new_resolver("eval", eval)


from matplotlib import rcParams

# Set global font properties
rcParams['font.family'] = 'helvetica'
def create_optimized_divergent_colormap():
    # Define the colors: soft red, white, soft green, and soft blue
    cdict = {
        'red':   [(0.0, 0.3, 0.3),  # Soft blue at the start
                  (0.5, 1.0, 1.0),  # White in the middle
                  (1.0, 0.8, 0.8)], # Soft red at the end
        'green': [(0.0, 0.4, 0.4),  # Soft blue at the start
                  (0.5, 1.0, 1.0),  # White in the middle
                  (1.0, 0.2, 0.2)], # Soft red at the end
        'blue':  [(0.0, 0.8, 0.8),  # Soft blue at the start
                  (0.5, 1.0, 1.0),  # White in the middle
                  (1.0, 0.3, 0.3)]  # Soft red at the end
    }
    # Create the colormap
    custom_cmap = LinearSegmentedColormap('OptimizedMap', segmentdata=cdict, N=256)
    return custom_cmap

# Create the custom colormap
cmap = create_optimized_divergent_colormap()

def plot_psychometric_data(psychometric_data, show=True, save=False, path=None):
    coherence_lvls_relevant = psychometric_data["coherence_lvls_relevant"]
    coherence_lvls_irrelevant = psychometric_data["coherence_lvls_irrelevant"]

    # invert cause of the axes running from the bottom to the top
    Motion_rght_prcntg = psychometric_data["motion"]["right_choice_percentage"]#[::-1, :]
    Motion_MSE = psychometric_data["motion"]["MSE"]

    Color_rght_prcntg = psychometric_data["color"]["right_choice_percentage"]
    Color_MSE = psychometric_data["color"]["MSE"]#[::-1, :]
    num_lvls = Color_rght_prcntg.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(1 * 140 * mm, 60 * mm))

    aspect_ratios = [len(coherence_lvls_irrelevant) / len(coherence_lvls_relevant),
                     len(coherence_lvls_relevant) / len(coherence_lvls_irrelevant)]
    for i, ctxt in enumerate(["Motion", "Color"]):
        im = axes[i].imshow(eval(f"{ctxt}_rght_prcntg"), cmap=cmap, interpolation="bicubic", aspect=aspect_ratios[i])

    ticks_irrelevant = ["-4", "0", "4"]
    tick_inds_irrelevant = []
    for tick in ticks_irrelevant:
        for c, coh in enumerate(coherence_lvls_relevant):
            if np.abs(float(tick) - coh) < 0.0001:
                tick_inds_irrelevant.append(c)
                break


    ticks_relevant = ["-1.06666666", "0", "1.06666666"]
    tick_inds_relevant = []
    for tick in ticks_relevant:
        for c, coh in enumerate(coherence_lvls_irrelevant):
            if np.abs(float(tick) - coh) < 0.0001:
                tick_inds_relevant.append(c)
                break
    ticks_relevant = ["-1", "0", "1"]

    axes[0].set_xticks(tick_inds_relevant)
    axes[0].set_xticklabels(ticks_relevant, rotation=0)

    axes[0].set_yticks(tick_inds_irrelevant)
    axes[0].set_yticklabels(ticks_irrelevant)

    axes[1].set_xticks(tick_inds_irrelevant)
    axes[1].set_xticklabels(ticks_irrelevant, rotation=0)

    axes[1].set_yticks(tick_inds_relevant)
    axes[1].set_yticklabels(ticks_relevant)

    # for i in range(2):
    #     axes[i].set_xticks(tick_inds)
    #     axes[i].set_xticklabels(ticks, rotation=90)

    if show:
        plt.show()
    if save:
        fig.savefig(path, dpi=300, transparent=True, bbox_inches='tight')
        path_png = path.split(".pdf")[0] + ".png"
        fig.savefig(path_png, dpi=300, transparent=True, bbox_inches='tight')

    plt.close(fig)
    return None


n_nets = 10
taskname = "CDDM"
show = False
save = True
@hydra.main(version_base="1.3", config_path=f"../../configs/task", config_name=f'{taskname}')
def analyze_behavior(cfg):
    file_str = os.path.join(cfg.task.paths.RNN_dataset_path, f"{taskname}_top30.pkl")
    data_save_folder = cfg.task.paths.fixed_points_data_folder
    img_save_folder = cfg.task.paths.img_folder

    # defining the task
    task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.task.dt)
    task = hydra.utils.instantiate(task_conf)
    coh_bounds = (-4, 4)
    DF_dict = {}
    connectivity_dict = {}
    for activation_name in ["relu"]:
        for constrained in [True]:
            activation_slope = eval(f"cfg.task.dataset_filtering_params.{activation_name}_filters.activation_slope")
            filters = {"activation_name": ("==", activation_name),
                       "activation_slope": ("==", activation_slope),
                       "RNN_score": ("<=", eval(f"cfg.task.dataset_filtering_params.{activation_name}_filters.RNN_score_filter")),
                       "constrained": ("==", constrained),
                       "lambda_r": (">=", eval(f"cfg.task.dataset_filtering_params.{activation_name}_filters.lambda_r"))}

            DF_dict[activation_name] = get_dataset(file_str, filters)[:n_nets]

            connectivity_dict[activation_name] = {}
            connectivity_dict[activation_name]["inp"] = DF_dict[activation_name]["W_inp_RNN"].tolist()
            connectivity_dict[activation_name]["rec"] = DF_dict[activation_name]["W_rec_RNN"].tolist()
            connectivity_dict[activation_name]["out"] = DF_dict[activation_name]["W_out_RNN"].tolist()
            folder_names = DF_dict[activation_name]["folder"].tolist()
            for i in range(n_nets):
                print(folder_names[i])
                W_inp = connectivity_dict[activation_name]["inp"][i]
                W_rec = connectivity_dict[activation_name]["rec"][i]
                W_out = connectivity_dict[activation_name]["out"][i]

                N = W_inp.shape[0]
                net_params = {"N": N,
                              "dt": cfg.task.dt,
                              "tau": cfg.task.tau,
                              "activation_name": activation_name,
                              "activation_slope": activation_slope,
                              "W_inp": W_inp,
                              "W_rec": W_rec,
                              "W_out": W_out,
                              "bias_rec": None,
                              "y_init": np.zeros(N)}
                rnn = RNN_numpy(**net_params)


                mask = np.concatenate([np.arange(100), 200 + np.arange(100)])
                pa = PerformanceAnalyzerCDDM(rnn)
                pa.calc_psychometric_data(task, mask,
                                          num_levels=31,
                                          coh_bounds=coh_bounds,
                                          num_repeats=7,
                                          sigma_rec=0.03,
                                          sigma_inp=0.03)

                path = os.path.join(img_save_folder,
                                    f"cropped_psychometric_plot_{activation_name}_constrained={constrained}_net={i}.pdf")

                cropped_psychometric_data = pa.psychometric_data
                inds = np.where(np.abs(np.array(cropped_psychometric_data["coherence_lvls"])) < 1.1)[0]
                cropped_psychometric_data["coherence_lvls_relevant"] = cropped_psychometric_data["coherence_lvls"]
                cropped_psychometric_data["coherence_lvls_irrelevant"] = [cropped_psychometric_data["coherence_lvls"][i] for i in inds]
                cropped_psychometric_data["motion"]["right_choice_percentage"] = cropped_psychometric_data["motion"]["right_choice_percentage"][:, inds]
                cropped_psychometric_data["motion"]["MSE"] = cropped_psychometric_data["motion"]["MSE"][:, inds]
                # cropped_psychometric_data["motion"]["coherence_levels_x"] = [cropped_psychometric_data["coherence_lvls"][i] for i in inds]
                # cropped_psychometric_data["motion"]["coherence_levels_y"] = cropped_psychometric_data["coherence_lvls"]


                cropped_psychometric_data["color"]["right_choice_percentage"] = cropped_psychometric_data["color"]["right_choice_percentage"][inds, :]
                cropped_psychometric_data["color"]["MSE"] = cropped_psychometric_data["color"]["MSE"][inds, :]
                # cropped_psychometric_data["color"]["coherence_levels_x"] = cropped_psychometric_data["coherence_lvls"]
                # cropped_psychometric_data["color"]["coherence_levels_y"] = [cropped_psychometric_data["coherence_lvls"][i] for i in inds]
                plot_psychometric_data(cropped_psychometric_data, show, save=save, path=path)

                plt.close()
                # colors = ["red", "blue", "green", "magenta", "cyan", "purple"]
                # labels = ["Right", "Left"]
                # fig, ax = plt.subplots(inputs.shape[-1], 1, figsize = (3,2))
                # plt.suptitle(f"Activation function: {activation_name}")
                # for j in range(outputs.shape[-1]):
                #     for k in range(outputs.shape[0]):
                #         ax[j].plot(outputs[k, :, j].T,
                #                    color=colors[k],
                #                    label=labels[k],
                #                    linewidth=2)
                #         ax[j].set_ylim([-0.1, 1.3])
                #         ax[j].axhline(0, linestyle='--')
                # ax[0].legend(loc="upper left")
                # path = os.path.join(img_save_folder, f"behavior_{taskname}_{activation_name}_{i}.pdf")
                # plt.savefig(path, bbox_inches='tight', dpi=300, transparent=True)
                # plt.show()

if __name__ == '__main__':
    analyze_behavior()
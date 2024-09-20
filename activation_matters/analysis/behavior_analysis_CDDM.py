import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from trainRNNbrain.training.training_utils import prepare_task_arguments

from activation_matters.plots.ploting_utils import plot_psychometric_data
from activation_matters.utils.feautre_extraction_utils import get_dataset
from style.style_setup import set_up_plotting_styles

np.set_printoptions(suppress=True)
import os
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.analyzers.DynamicSystemAnalyzer import *
from trainRNNbrain.analyzers.PerformanceAnalyzer import *

os.system('python ../../style/style_setup.py')
import hydra

OmegaConf.register_new_resolver("eval", eval)
n_nets = 1
taskname = "CDDM"
show = True
save = True
@hydra.main(version_base="1.3", config_path=f"../../configs/task", config_name=f'{taskname}')
def analyze_behavior(cfg):
    # set_up_plotting_styles(cfg.task.paths.style_path)
    file_str = os.path.join(cfg.task.paths.RNN_datasets_path, f"{taskname}.pkl")
    data_save_folder = cfg.task.paths.fixed_points_data_folder
    img_save_folder = cfg.task.paths.img_folder

    # defining the task
    task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.task.dt)
    task = hydra.utils.instantiate(task_conf)
    coh_bounds = (-4, 4)
    DF_dict = {}
    connectivity_dict = {}
    for activation_name in ["relu", "tanh"]:
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

            for i in range(n_nets):
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


                # if taskname == "CDDM":
                #     labels = ["Right", "Left"]
                #     input_stream1, _ = task.generate_input_target_stream(context="motion",
                #                                                         motion_coh=-0.25,
                #                                                         color_coh=3)
                #     input_stream2, _ = task.generate_input_target_stream(context="color",
                #                                                         motion_coh=3,
                #                                                         color_coh=-0.25)
                #     inputs = np.stack((input_stream1, input_stream2), axis=2)

                # inputs, targets, conditions =  task.get_batch()
                # rnn.run(inputs)
                # outputs = rnn.get_output()

                # get Psychometric plots!
                # mask_params = [evaluate_expression(param, RNN_config["task"]) for param in
                #                RNN_config["task"]['mask_params']]
                # mask = get_training_mask(mask_params, dt=dt)

                mask = np.concatenate([np.arange(100), 200 + np.arange(100)])
                pa = PerformanceAnalyzerCDDM(rnn)
                pa.calc_psychometric_data(task, mask,
                                          num_levels=31,
                                          coh_bouds=coh_bounds,
                                          num_repeats=7,
                                          sigma_rec=0.03,
                                          sigma_inp=0.03)

                path = os.path.join(img_save_folder,
                                    f"psychometric_plot_{activation_name}_constrained={constrained}_net={i}.pdf")
                plot_psychometric_data(pa.psychometric_data, show, save=save, path=path)

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
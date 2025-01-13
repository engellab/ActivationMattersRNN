import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from trainRNNbrain.training.training_utils import prepare_task_arguments
from activation_matters.plots.ploting_utils import plot_psychometric_data
import pickle
np.set_printoptions(suppress=True)
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.analyzers.PerformanceAnalyzer import *
import hydra

OmegaConf.register_new_resolver("eval", eval)

n_nets = 1
taskname = "CDDM"
show = True
save = False

@hydra.main(version_base="1.3", config_path=f"../../configs/task", config_name=f'{taskname}')
def analyze_behavior(cfg):
    taskname = cfg.task.taskname
    dataset_path = os.path.join(f"{cfg.task.paths.RNN_dataset_path}", f"{taskname}_top30.pkl")
    dataset = pickle.load(open(dataset_path, "rb"))
    img_save_folder = os.path.join(cfg.task.paths.img_folder)

    # defining the task
    task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.task.dt)
    task = hydra.utils.instantiate(task_conf)
    coh_bounds = (-4, 4)
    DF_dict = {}
    connectivity_dict = {}
    for activation_name in ["relu", "tanh"]:
        for constrained in [True]:
            connectivity_dict[activation_name] = {}
            connectivity_dict[activation_name]["inp"] = dataset[activation_name][f"Dale={constrained}"]["W_inp_RNN"].tolist()
            connectivity_dict[activation_name]["rec"] = dataset[activation_name][f"Dale={constrained}"]["W_rec_RNN"].tolist()
            connectivity_dict[activation_name]["out"] = dataset[activation_name][f"Dale={constrained}"]["W_out_RNN"].tolist()

            for i in range(n_nets):
                W_inp = connectivity_dict[activation_name]["inp"][i]
                W_rec = connectivity_dict[activation_name]["rec"][i]
                W_out = connectivity_dict[activation_name]["out"][i]

                activation_slope = dataset[activation_name][f"Dale={constrained}"]["activation_slope"].tolist()[0]

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
                                          coh_bouds=coh_bounds,
                                          num_repeats=7,
                                          sigma_rec=0.03,
                                          sigma_inp=0.03)

                path = os.path.join(img_save_folder,
                                    f"psychometric_plot_{activation_name}_constrained={constrained}_net={i}.pdf")
                plot_psychometric_data(pa.psychometric_data, show, save=save, path=path)

                plt.close()


if __name__ == '__main__':
    analyze_behavior()
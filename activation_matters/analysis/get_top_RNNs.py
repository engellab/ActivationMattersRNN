import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from trainRNNbrain.training.training_utils import prepare_task_arguments

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
n_nets = 10
taskname = "MemoryNumber"
show = False
@hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'{taskname}_conf')
def get_top_RNNs(cfg):
    set_up_plotting_styles(cfg.paths.style_path)
    file_str = os.path.join(cfg.paths.RNN_datasets_path, f"{taskname}.pkl")
    DF_dict = {}
    connectivity_dict = {}
    parent_folders = []
    subfolders = []
    for activation_name in ["relu", "sigmoid", "tanh"]:
        for constrained in [True, False]:
            activation_slope = eval(f"cfg.dataset_filtering_params.{activation_name}_filters.activation_slope")
            filters = {"activation_name": ("==", activation_name),
                       "activation_slope": ("==", activation_slope),
                       "RNN_score": ("<=", eval(f"cfg.dataset_filtering_params.{activation_name}_filters.RNN_score_filter")),
                       "constrained": ("==", constrained),
                       "lambda_r": (">=", eval(f"cfg.dataset_filtering_params.{activation_name}_filters.lambda_r"))}
            DF_dict[activation_name] = get_dataset(file_str, filters)[:n_nets]
            paths = DF_dict[activation_name]["folder"].tolist()
            for path in paths:
                parent_folders.append(str(path).split("/")[0])
                subfolders.append(str(path).split("/")[1])
    print(parent_folders)
    print(subfolders)

if __name__ == '__main__':
    get_top_RNNs()
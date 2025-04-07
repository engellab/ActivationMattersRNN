import numpy as np
import pickle
import hydra
from omegaconf import OmegaConf
from trainRNNbrain.training.training_utils import prepare_task_arguments
from activation_matters.plots.style.style_setup import set_up_plotting_styles
import os

# @hydra.main(version_base="1.3", config_path=f"../../configs/task/", config_name=f'{taskname}')
@hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'base')
def test_loading(cfg: OmegaConf):
    taskname = cfg.task.taskname
    n_nets = cfg.n_nets
    dataSegment = cfg.dataSegment
    set_up_plotting_styles(cfg.paths.style_path)
    aux_datasets_folder = os.path.join(cfg.paths.auxilliary_datasets_path, taskname)
    img_save_folder = os.path.join(cfg.paths.img_folder, taskname)

    # defining the task
    task_conf = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.task.dt)
    task = hydra.utils.instantiate(task_conf)
    if taskname == "CDDM":
        task.coherences = np.array([-1, -0.5, 0, 0.5, 1.0])
    if hasattr(task, 'random_window'):
        task.random_window = 0
    task.seed = 0  # for consistency
    inputs, targets, conditions = task.get_batch()

    print(dataSegment)

    data_dict1 = pickle.load(
        open(os.path.join(aux_datasets_folder, f"trajectories_data_top{n_nets}.pkl"), "rb+"))
    data_dict2 = pickle.load(
        open(os.path.join(aux_datasets_folder, f"trajectories_data_bottom{n_nets}.pkl"), "rb+"))
    x = 1


if __name__ == '__main__':
    test_loading()


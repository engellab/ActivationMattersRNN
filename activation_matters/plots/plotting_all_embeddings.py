import numpy as np
np.set_printoptions(suppress=True)

from activation_matters.plots.plotting_MDS_embedding_fixed_point_configurations import plot_fixed_point_configurations
from activation_matters.plots.plotting_MDS_embedding_single_unit_selectivity import plot_MDS_of_selectivities
from activation_matters.plots.plotting_MDS_embedding_trajectories import plot_MDS_of_trajectories
from activation_matters.plots.plotting_MDS_embedding_trajectory_endpoints import plot_trajectory_endpoints
import hydra
from omegaconf import OmegaConf
import os
OmegaConf.register_new_resolver("eval", eval)

os.environ["HYDRA_FULL_ERROR"] = "1"
@hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'base')
def plot_all_embeddings(cfg):
    print(f"Running analysis on {cfg.task.taskname}")
    print(f"{cfg.task.taskname}: Plotting embedding of trajectories...")
    plot_MDS_of_trajectories(cfg)
    print(f"{cfg.task.taskname}: Plotting embedding of single-unit selectivity...")
    plot_MDS_of_selectivities(cfg)
    print(f"{cfg.task.taskname}: Plotting embedding of fixed points...")
    plot_fixed_point_configurations(cfg)
    print(f"{cfg.task.taskname}: Plotting embedding of trajectory endpoints...")
    plot_trajectory_endpoints(cfg)

if __name__ == '__main__':
    plot_all_embeddings()


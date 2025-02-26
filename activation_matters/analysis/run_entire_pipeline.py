import numpy as np
np.set_printoptions(suppress=True)
from activation_matters.analysis.computing_fp import *
from activation_matters.analysis.trajectory_analysis import *
from activation_matters.analysis.selectivity_analysis import *
from activation_matters.analysis.fixed_point_analysis import *
from activation_matters.analysis.trajectory_endpoints_analysis import *
import hydra
from omegaconf import OmegaConf
import os
OmegaConf.register_new_resolver("eval", eval)

os.environ["HYDRA_FULL_ERROR"] = "1"
@hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'base')
def run_entire_pipeline(cfg):
    print(f"Running analysis on {cfg.task.taskname}")
    print(f"{cfg.task.taskname}: Analyzing trajectories...")
    trajectory_analysis(cfg)
    print(f"{cfg.task.taskname}: Analyzing single-unit selectivity...")
    selectivity_analysis(cfg)
    print(f"{cfg.task.taskname}: Computing fixed points...")
    computing_fp(cfg)
    print(f"{cfg.task.taskname}: Analyzing fixed points...")
    fixed_point_analysis(cfg)
    print(f"{cfg.task.taskname}: Analyzing trajectory endpoints...")
    trajectory_endpoints_analysis(cfg)

if __name__ == '__main__':
    run_entire_pipeline()


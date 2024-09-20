import numpy as np
np.set_printoptions(suppress=True)
import os
from activation_matters.utils.feautre_extraction_utils import *
import warnings
from omegaconf import OmegaConf
import hydra
from pathlib import Path
warnings.filterwarnings('ignore')
OmegaConf.register_new_resolver("eval", eval)

taskname = 'GoNoGo'
@hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'{taskname}_conf')
def extract_features_from_trajectories(cfg):
    RNN_datasets_str = os.path.join(cfg.paths.RNN_datasets_path, f"{taskname}.pkl")
    for activation_name in ["relu", "sigmoid", "tanh"]:
        for constrained in [True, False]:
            score = eval(f"cfg.dataset_filtering_params.{activation_name}_filters.RNN_score_filter")
            lmbd_r = eval(f"cfg.dataset_filtering_params.{activation_name}_filters.lambda_r")
            activation_slope = eval(f"cfg.dataset_filtering_params.{activation_name}_filters.activation_slope")
            filters = {"activation_name": ("==", activation_name),
                       "activation_slope":  ("==", activation_slope),
                       "RNN_score": ("<=", score),
                       "constrained": ("==", constrained),
                       "lambda_r": (">=", lmbd_r)}
            df_filtered = get_dataset(RNN_datasets_str, filters)

            print(f"activation={activation_name}; constrained={constrained}: "
                  f"there are {len(df_filtered)} RNNs with specified parameters")
            if len(df_filtered) == 0:
                raise ValueError("Empty dataframe! Relax the constraints")

            processing_params = {}
            if not (cfg.feature_extraction_params.Trajectory is None):
                processing_params["Trajectory"] = cfg.feature_extraction_params.Trajectory
            if not (cfg.feature_extraction_params.Connectivity is None):
                processing_params["Connectivity"] = cfg.feature_extraction_params.Connectivity

            nrn_type_importance_factor = eval(f"cfg.representations_analysis.{activation_name}_nets.nrn_type_importance_factor")
            connectivity_importance_factor = eval(f"cfg.representations_analysis.{activation_name}_nets.connectivity_importance_factor")
            feature_data = extract_features(dataframe=df_filtered,
                                            downsample_window=cfg.feature_extraction_params.downsample_window,
                                            processing_params=processing_params,
                                            connectivity_importance_factor=1.0,
                                            nrn_type_importance_factor=0.5)

            N = feature_data["Features"].shape[0]
            print(f"N extracted neurons: {N}")
            save_to_folder = cfg.paths.feature_datasets_folder
            if not os.path.exists(save_to_folder):
                Path(save_to_folder).mkdir(parents=True, exist_ok=False)
            file_path = (f"{save_to_folder}/{taskname}_{activation_name}_"
                         f"Dale={constrained}_"
                         f"typeif={nrn_type_importance_factor}_"
                         f"connif={connectivity_importance_factor}.pkl")
            pickle.dump(feature_data, open(file_path, 'wb+'))

if __name__ == '__main__':
    extract_features_from_trajectories()

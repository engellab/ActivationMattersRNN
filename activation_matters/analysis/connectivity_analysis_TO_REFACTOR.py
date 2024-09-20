from omegaconf import OmegaConf
import hydra
from activation_matters.utils.feautre_extraction_utils import reduce_dimensionality
from activation_matters.utils.get_intercluster_connectivity import get_intercluster_connectivity
from activation_matters.utils.graph_clustering import ModelSelector, CustomClusterer, get_best_clustering
from activation_matters.plots.ploting_utils import *
from itertools import chain
import os
from style.style_setup import set_up_plotting_styles


OmegaConf.register_new_resolver("eval", eval)
taskname = 'MemoryNumber'
@hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'{taskname}_conf')
def analyze_representations(cfg):
    set_up_plotting_styles(cfg.paths.style_path)
    save = True#cfg.representations_analysis.save_figures
    show = False #cfg.representations_analysis.show_figures

    for activation_name in ["relu", "sigmoid", "tanh"]:
        for constrained in [True, False]:
            print(f"Calculating number of clusters and inter-cluster connectivity for {taskname} {activation_name} Dale={constrained} RNNs:")
            nrn_type_if = eval(f"cfg.representations_analysis.{activation_name}_nets.nrn_type_importance_factor")
            conn_if = eval(f"cfg.representations_analysis.{activation_name}_nets.connectivity_importance_factor")

            path_to_feature_dataset = os.path.join(cfg.paths.feature_datasets_folder,
                                                   f"{taskname}_"
                                                   f"{activation_name}_"
                                                   f"Dale={constrained}_"
                                                   f"typeif={nrn_type_if}"
                                                   f"_connif={conn_if}.pkl")
            img_path = cfg.paths.img_folder
            file = open(path_to_feature_dataset, "rb+")
            Feature_data = pickle.load(file)

            Feature_array = Feature_data["Features"]
            feature_list = Feature_data["Features_by_RNN"]
            # Trajectories_Feature = Feature_data["Trajectory_Features"]

            W_inp_list = Feature_data["W_inp_list"]
            W_rec_list = Feature_data["W_rec_list"]
            W_out_list = Feature_data["W_out_list"]

            clusterer_params = {"max_iter": cfg.representations_analysis.max_iter,
                                "gamma": cfg.representations_analysis.gamma,
                                "increase_gamma": cfg.representations_analysis.increase_gamma}
            clusterer_bldr_fn = lambda n_clusters: CustomClusterer(n_clusters, **clusterer_params)
            model_selector = ModelSelector(clusterer_bldr_fn,
                                           scoring_method=cfg.representations_analysis.cluster_scoring_method,
                                           min_clusters=cfg.representations_analysis.min_clusters,
                                           max_clusters=cfg.representations_analysis.max_clusters,
                                           n_repeats=cfg.representations_analysis.n_repeats,
                                           selection_criteria=cfg.representations_analysis.selection_criteria,
                                           parallel=False)
            optimal_n_clusters = model_selector.select_n_clusters(feature_list=feature_list, W_recs=W_rec_list, visualize=False)
            print(f"Optimal number of clusters: {optimal_n_clusters}")

            clusterer = CustomClusterer(n_clusters=optimal_n_clusters, **clusterer_params)
            best_score, labels_list = get_best_clustering(clusterer=clusterer,
                                                          feature_list=feature_list,
                                                          W_recs=W_rec_list,
                                                          scoring_method=cfg.representations_analysis.cluster_scoring_method,
                                                          n_repeats=cfg.representations_analysis.n_repeats)
            labels_unraveled = list(chain.from_iterable(labels_list))
            ic_W_inp, ic_W_rec, ic_W_out = get_intercluster_connectivity(W_inp_list, W_rec_list, W_out_list, labels_list)
            print(f"Best clustering Score: {best_score}")
            Features_projected, features_list_projected = reduce_dimensionality(Feature_array=Feature_data["Features"],
                                                                                Feature_list=Feature_data["Features_by_RNN"],
                                                                                var_thr=0.99)
            # for axes in [(0, 1, 2), (0, 2, 3), (0, 3, 2), (1, 3, 0), (1, 2, 3)]:
            #     print(axes)
            #     plot_representations_2D(Trajectories_Feature, axes=axes[:2], show=show, save=save,
            #                             path=os.path.join(img_path, f"representations2D_axes={axes}_{activation_name}_constrained={constrained}.png"),
            #                             s=15, alpha=0.75)
            #     plot_representations_3D(Trajectories_Feature, axes=axes, show=show, save=save,
            #                             path=os.path.join(img_path, f"representations3D_axes={axes}_{activation_name}_constrained={constrained}.png"),
            #                             s=10, alpha=0.75)

            # for axes in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
            #     print(axes)
            #     plot_representations_3D(Trajectories_Feature, axes=axes, show=show, save=save,
            #                             path=os.path.join(img_path, f"representations3D_axes={axes}_{activation_name}_constrained={constrained}.png"),
            #                             s=15, alpha=0.75)

            # for axes in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
            #              (1, 0), (2, 0), (3, 0), (2, 1), (3, 1), (3, 2)]:
            #     print(axes)
            #     plot_representations_2D(Trajectories_Feature, axes=axes, show=show, save=save,
            #                             path=os.path.join(img_path, f"representations2D_axes={axes}_{activation_name}_constrained={constrained}.png"),
            #                             s=15, alpha=0.75)

            plot_RNN_distribution(labels_list, show=show, save=save, path=os.path.join(img_path, f"distribution_{activation_name}_constrained={constrained}.pdf"))
            plot_feature_array(Features_projected, labels_unraveled, show=show, save=save, path=os.path.join(img_path, f"features_{activation_name}_constrained={constrained}_3D.pdf"))
            plot_intercluster_connectivity(ic_W_inp=ic_W_inp,
                                           ic_W_rec=ic_W_rec,
                                           ic_W_out=ic_W_out,
                                           labels_unraveled=labels_unraveled,
                                           th=cfg.representations_analysis.thr,
                                           show=show, save=save,
                                           path=os.path.join(img_path, f"intercluster_connectivity_{activation_name}_constrained={constrained}.pdf"))

if __name__ == '__main__':
    analyze_representations()


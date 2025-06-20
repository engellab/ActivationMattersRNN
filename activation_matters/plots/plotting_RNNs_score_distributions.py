import numpy as np
import os
import hydra
import pickle
from matplotlib import pyplot as plt
# plt.rcParams['text.usetex'] = True
import pandas as pd
from activation_matters.plots.style.style_setup import set_up_plotting_styles
from copy import deepcopy
import seaborn as sns
mm = 1/25.4

def merge_dictionaries(dict1, dict2):
    dict_res = deepcopy(dict1)
    for key in dict2.keys():
        for subkey in dict2[key].keys():
            dict_res[key][subkey] = pd.concat([dict1[key][subkey], dict2[key][subkey]])
    return dict_res

@hydra.main(version_base="1.3", config_path=f"../../configs", config_name=f'base')
def plot_r2_distributions(cfg):
    show = False
    save = True
    taskname = cfg.task.taskname
    print(taskname)
    set_up_plotting_styles(cfg.paths.style_path)
    RNN_dataset_path = os.path.join(cfg.paths.RNN_dataset_path)
    img_folder = os.path.join(cfg.paths.img_folder, taskname)

    # data_dict = (pickle.load(open(os.path.join(RNN_dataset_path, f"{taskname}_top50.pkl"), "rb+")))
    # data_dict = (pickle.load(open(os.path.join(RNN_dataset_path, f"{taskname}_bottom50.pkl"), "rb+")))
    data_top = pickle.load(open(os.path.join(RNN_dataset_path, f"{taskname}_top50.pkl"), "rb+"))
    data_bot = pickle.load(open(os.path.join(RNN_dataset_path, f"{taskname}_bottom50.pkl"), "rb+"))
    data_dict = merge_dictionaries(data_top, data_bot)
    dict_list = []
    for activation in ["relu", "sigmoid", "tanh"]:
        for constrained_key in ["Dale=True", "Dale=False"]:
            dict_list.append(pd.DataFrame(data_dict[activation][constrained_key]))
            # dict_list.append(pd.DataFrame(dict_bottom[activation][constrained_key]))
    df = pd.concat(dict_list, axis=0)
    R2_dict = {}
    for activation in ["relu", "sigmoid", "tanh"]:
        R2_dict[activation] = {}
        for constrained in [True, False]:
            R2_dict[activation][f"Dale={constrained}"] = []

    for row in df.itertuples():
        activation = row.activation_name
        constrained = row.constrained
        R2_dict[activation][f"Dale={constrained}"].append(row.R2_score)

    # Define colors for each condition
    colors = {
        ("relu", True): "red",
        ("relu", False): "orange",
        ("sigmoid", True): "blue",
        ("sigmoid", False): "deepskyblue",
        ("tanh", True): "green",
        ("tanh", False): "lightgreen",
    }

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(60*mm, 60*mm))
    suptitle_map = {"CDDM" : "CDDM", "GoNoGo": "Go/NoGo", "MemoryNumber": "Memory Number"}
    legend_map = {""}
    fig.suptitle(suptitle_map[taskname]) #Distribution of r² Scores,
    kw = 0.2
    cnt = 0
    lb = np.inf
    ub = -np.inf
    for activation in R2_dict:
        for constrained in R2_dict[activation]:
            # Extract the data
            data = np.array(R2_dict[activation][constrained]).flatten()
            print(activation, constrained, np.round(100 * np.min(data),1), np.round(100 * np.mean(data),1), np.round(100 * np.std(data), 1))

            lb = np.min(data) if lb >= np.min(data) else lb
            ub = np.max(data) if ub <= np.max(data) else ub

            # Determine the color
            constrained_bool = constrained == "Dale=True"
            color = colors[(activation, constrained_bool)]
            # Plot the KDE
            if constrained_bool ==True:
                Dale_tag = 'Dale'
            else:
                Dale_tag = 'no Dale'

            bin_width = 0.0025
            bins = np.arange(lb, ub + bin_width, bin_width)
            axes[cnt // 2].hist(data, bins=bins, color=color, alpha=0.8,
                                edgecolor='k', linewidth=0.1, label=f"{activation}, {Dale_tag}")
            # sns.kdeplot(data, fill = True, alpha = 0.1, ax=axes[cnt//2], common_norm=False,
            #             bw_adjust = kw, label=f"{activation}, Dale={constrained_bool}", color=color, linewidth=0.5)
            axes[cnt // 2].spines['top'].set_visible(False)
            axes[cnt // 2].spines['right'].set_visible(False)
            axes[cnt // 2].spines['bottom'].color = 'gray'
            if taskname == "MemoryNumber":
                axes[cnt // 2].legend(loc='upper left', ncol=1, bbox_to_anchor=(0.00, 1.05))

            axes[cnt // 2].set_ylabel(None)
            # axes[cnt // 2].set_yticks([])
            # axes[cnt // 2].set_yticklabels([])
            axes[cnt // 2].set_yticks([np.floor(axes[cnt // 2].get_ylim()[0]), np.floor(axes[cnt // 2].get_ylim()[-1])])
            cnt += 1
    for i in range(cnt//2):
        axes[i].set_xlim([lb - 0.0025, ub + 0.0025])
        axes[i].set_xticks([np.round(axes[i].get_xlim()[0], 2), np.round(axes[i].get_xlim()[-1],2)])
        if i != 2:
            axes[i].set_xticklabels([])
        # axes[-1].set_xlabel(r"$\textit{R}^2$", loc='center')
    if taskname == "CDDM":
        # axes[1].set_ylabel(r"$\mathrm{No.}$ RNNs")
        axes[1].set_ylabel(r"Number of RNNs", labelpad=5)
    path = os.path.join(img_folder, f"R2 distribution {taskname}.pdf")
    if save:
        print("Saving figure to", path)
        plt.savefig(path, dpi=300, transparent=True, bbox_inches='tight')
        path_png = path.split(".pdf")[0] + ".png"
        plt.savefig(path_png, dpi=300, transparent=True, bbox_inches='tight')
    if show: plt.show()
    print(lb)

if __name__ == '__main__':
    plot_r2_distributions()



from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from activation_matters.utils.utils import *
from itertools import chain
from sklearn.decomposition import PCA
import os
import numpy as np
mm = 1/25.4


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


def normalize_color(color):
    if type(color) == str:
        return color
    if max(color) <= 1.0:
        return color
    return [c / 255.0 for c in color]

def plot_projected_trajectories(trajectories_projected,
                                legend,
                                axes=(0, 1),
                                save=False, path=None,
                                line_colors=None,
                                point_colors=None,
                                markers=None,
                                linewidth=0.75,
                                linewidth_edge=0.1,
                                markersize = 25,
                                alpha=0.9,
                                n_dim=2,
                                show=True):
    if n_dim == 2:
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    else:
        fig = plt.figure(figsize=(2, 2))
        ax = plt.axes(projection='3d')
        ax.view_init(45, -135)
    for trial in range(trajectories_projected.shape[-1]):
        trajectory = trajectories_projected[:, :, trial]
        if not (point_colors is None):
            distances = np.sqrt(np.sum([np.diff(trajectory[i])**2 for i in range(n_dim)], axis = 0))
            cumulative_distance = np.insert(np.cumsum(distances), 0, 0)
            num_points = 8
            equidistant_cumulative_distance = np.linspace(0, cumulative_distance[-1], num_points)
            interp = [interp1d(cumulative_distance, trajectory[axes[i], :], kind='linear') for i in range(n_dim)]
            interpolated_trajectory = [interp[i](equidistant_cumulative_distance) for i in range(n_dim)]
            ax.scatter(*[interpolated_trajectory[i] for i in range(n_dim)],
                    color=point_colors[trial], facecolors=point_colors[trial],
                    linewidth=linewidth_edge, marker=markers[trial], alpha=alpha, s=markersize, edgecolors='k')

        ax.plot(*[trajectory[axes[i], :,] for i in range(n_dim)],
                color=line_colors[trial],
                linewidth=linewidth, alpha=alpha)

    axes_names = ["x", "y", "z"][:n_dim]
    for a, axis in enumerate(axes_names):
        for postfix in ["min", "max"]:
            exec(f"{axis}_{postfix} = np.{postfix}(trajectories_projected[{axes[a]}, ...])")
        exec(f"{axis}_range = 1.2 * np.abs(({axis}_max - {axis}_min))")
        exec(f"{axis}_mean = ({axis}_min + {axis}_max) / 2")
    for a, axis in enumerate(axes_names):
        exec(f"{axis}_min_new = ({axis}_mean - {axis}_range / 2)")
        exec(f"{axis}_max_new = ({axis}_mean + {axis}_range / 2)")
        exec(f"ax.set_{axis}lim([{axis}_min_new, {axis}_max_new])")
        exec(f"ax.set_{axis}ticks([])")
        exec(f"ax.set_{axis}ticklabels([])")

    # plt.suptitle(legend)
    if show:
        plt.show()
    if save:
        fig.savefig(path, transparent=True, dpi=300)
    plt.close(fig)
    return None

def plot_similarity_matrix(Mat, save=False, path=None, show=True):
    np.fill_diagonal(Mat, np.nanmean(Mat))
    Mat = np.nan_to_num(Mat, np.nanmean(Mat))
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im = ax.imshow(Mat, cmap=cmap, vmin=np.nanquantile(Mat, 0.25), vmax=np.nanquantile(Mat, 0.75))
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_ticks([np.quantile(Mat, 0.25), np.nanquantile(Mat, 0.75)])
    ax.set_yticks([])
    if save:
        fig.savefig(path, dpi=300, transparent=True, bbox_inches='tight')
        path_png = path.split(".pdf")[0] + ".png"
        fig.savefig(path_png, dpi=300, transparent=True, bbox_inches='tight')
    if show:
        plt.show()
    return None

def plot_embedding(embedding, inds_list, legends, colors, hatch=None, markers=None,
                   show_legends=False, save=False, path=None, show=True, ncols=3):
    fig = plt.figure(figsize=(150 * mm, 90 * mm))
    ax = fig.add_subplot(1, 1, 1)

    if hatch is None:
        hatch = ['' for i in range(len(inds_list))]
    if markers is None:
        markers = ['o' for i in range(len(inds_list))]

    for i in range(len(inds_list)):
        plt.scatter(embedding[inds_list[i], 0], embedding[inds_list[i], 1],
                    edgecolor='black',
                    color=colors[i],
                    label=legends[i],
                    hatch=hatch[i],
                    marker=markers[i],
                    linewidth=0.2,
                    s=30,
                    alpha=0.9)

    for a, axis in enumerate(["x", "y"]):
        for postfix in ["min", "max"]:
            exec(f"{axis}_{postfix} = np.{postfix}(embedding[:, {a}])")
        exec(f"{axis}_range = 1.15 * ({axis}_max - {axis}_min)")
        exec(f"{axis}_mean = ({axis}_min + {axis}_max) / 2")
    for a, axis in enumerate(["x", "y"]):
        exec(f"{axis}_min_new = ({axis}_mean - {axis}_range / 2)")
        exec(f"{axis}_max_new = ({axis}_mean + {axis}_range / 2)")
        exec(f"ax.set_{axis}lim([{axis}_min_new, {axis}_max_new])")
        exec(f"ax.set_{axis}ticklabels([])")
        exec(f"ax.set_{axis}ticks([])")

    plt.tight_layout()
    if show_legends:
        plt.legend(ncol=ncols, loc="upper center", fontsize=6)
    if save:
        plt.savefig(path, dpi=300, transparent=True, bbox_inches='tight')
        path_png = path.split(".pdf")[0] + ".png"
        plt.savefig(path_png, dpi=300, transparent=True, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return None

def plot_representations(F, axes=(0, 1), labels=None, show=True, save=False, path=None, s=10, alpha=0.5, n_dim=2):
    # Get the 'tab20' colormap
    cmap = plt.get_cmap('tab20')
    # Generate a list of 18 colors
    colors = [cmap(i) for i in range(20)]
    colors = [colors[(i + 6) % len(colors)] for i in range(len(colors))]
    if not (labels is None):
        n_clusters = len(np.unique(labels))
    else:
        n_clusters = 1
        labels = np.zeros(F.shape[0])
    if n_clusters == 1:
        colors = [[0.8, 0.2, 0.3]]

    if n_dim == 2:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    elif n_dim == 3:
        fig = plt.figure(figsize=(2, 2))
        ax = plt.axes(projection='3d')
        ax.view_init(45, -135)

    for i in range(n_clusters):
        inds = np.where(np.array(labels) == i)[0]
        ax.scatter(*[F[inds, axes[d]] for d in range(n_dim)],
                   edgecolors='k', color=colors[i], s=s, alpha=alpha, linewidths=0.1)

    # ax.set_title("Neural trajectory representations", loc='center')
    # ax.set_xlabel("PC1")
    # ax.set_ylabel("PC2")
    axes_names = ["x", "y", "z"][:n_dim]
    for a, axis in enumerate(axes_names):
        for postfix in ["min", "max"]:
            exec(f"{axis}_{postfix} = np.{postfix}(F[:, {axes[a]}])")
        exec(f"{axis}_range = 1.2 * np.abs(({axis}_max - {axis}_min))")
        exec(f"{axis}_mean = ({axis}_min + {axis}_max) / 2")
    for a, axis in enumerate(axes_names):
        exec(f"{axis}_min_new = ({axis}_mean - {axis}_range / 2)")
        exec(f"{axis}_max_new = ({axis}_mean + {axis}_range / 2)")
        exec(f"ax.set_{axis}lim([{axis}_min_new, {axis}_max_new])")
        exec(f"ax.set_{axis}ticks([])")
        exec(f"ax.set_{axis}ticklabels([])")
    if save:
        plt.savefig(path, dpi=300, transparent=True, bbox_inches='tight')
        path_png = path.split(".pdf")[0] + ".png"
        plt.savefig(path_png, dpi=300, transparent=True, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    return None

def plot_stimuli_representations(PCA_stimuli,
                                 face_colors,
                                 edge_colors,
                                 markers,
                                 s=50,
                                 alpha=0.7,
                                 show=True,
                                 save=False,
                                 path=None,
                                 n_dim=2):
    if n_dim == 3:
        fig = plt.figure(figsize=(2, 2))
        ax = plt.axes(projection='3d')
        ax.view_init(45, -135)
    elif n_dim == 2:
        fig, ax = plt.subplots(figsize=(2, 2))
    for marker in set(markers):
        inds = np.where(np.array(markers) == marker)[0]
        ax.scatter(*[PCA_stimuli[inds, i] for i in range(n_dim)],
                   color=[face_colors[i] for i in inds],
                   edgecolor=[edge_colors[i] for i in inds] , marker=marker, s=s, alpha=alpha, linewidths=1)
    axes_names = ["x", "y", "z"][:n_dim]
    for a, axis in enumerate(axes_names):
        for postfix in ["min", "max"]:
            exec(f"{axis}_{postfix} = np.{postfix}(PCA_stimuli[:, {a}])")
        exec(f"{axis}_range = 1.2 * ({axis}_max - {axis}_min)")
        exec(f"{axis}_mean = ({axis}_min + {axis}_max) / 2")
    for a, axis in enumerate(["x", "y"]):
        exec(f"{axis}_min_new = ({axis}_mean - {axis}_range / 2)")
        exec(f"{axis}_max_new = ({axis}_mean + {axis}_range / 2)")
        exec(f"ax.set_{axis}lim([{axis}_min_new, {axis}_max_new])")
        exec(f"ax.set_{axis}ticks([])")
        exec(f"ax.set_{axis}ticklabels([])")
    if show:
        plt.show()
    if save:
        fig.savefig(path, dpi=300, transparent=True, bbox_inches='tight')
        path_png = path.split(".pdf")[0] + ".png"
        plt.savefig(path_png, dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
    return False

def plot_intercluster_connectivity(ic_W_inp, ic_W_rec, ic_W_out, labels_unraveled, th=0.15, show=True, save=False, path=None):
    idxs = get_ordering(ic_W_inp, th = th)
    ic_W_inp = permute_input_matrix(ic_W_inp, idxs)
    ic_W_rec = permute_recurrent_matrix(ic_W_rec, idxs)
    ic_W_out = permute_output_matrix(ic_W_out, idxs)
    labels_permuted = np.array(idxs)
    labels_unraveled_permuted = [labels_permuted[lbl] for lbl in labels_unraveled]

    value_thr = 0.05
    num_inputs = ic_W_inp.shape[1]
    num_outputs = ic_W_out.shape[0]
    fig, ax = plt.subplots(3, 1, figsize=(4, 5), constrained_layout=False,
                           gridspec_kw={'height_ratios': [1, ic_W_rec.shape[0] / num_inputs, num_outputs / num_inputs]})
    matrix = ic_W_inp.T
    ax[0].imshow(matrix, cmap=cmap, vmin=-0.5, vmax=0.5, aspect='equal')
    # Add text annotations for each pixel
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.abs(matrix[i, j]) >= value_thr:
                ax[0].text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='black')

    matrix = ic_W_rec
    ax[1].imshow(matrix, cmap=cmap, vmin=-1, vmax=1, aspect='equal')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.abs(matrix[i, j]) >= value_thr:
                ax[1].text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='black')
    matrix = ic_W_out
    ax[2].imshow(matrix, cmap=cmap, vmin=-1, vmax=1, aspect='equal')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.abs(matrix[i, j]) >= value_thr:
                ax[2].text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='black')
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[2].set_xticks(np.arange(ic_W_rec.shape[0]))
    ax[1].set_yticks(np.arange(ic_W_rec.shape[0]))

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.tight_layout()
    if save:
        plt.savefig(path, dpi=300, transparent=True, bbox_inches='tight')
        path_png = path.split(".pdf")[0] + ".png"
        plt.savefig(path_png, dpi=300, transparent=True, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    return None


def plot_matrices(W_inp, W_rec, W_out, th=0.05, show=True, save=False, path=None):
    inds_sorted = get_ordering(np.hstack([W_inp, W_out.T]), th=th)
    W_inp_permuted = W_inp[inds_sorted, :]
    W_out_permuted = W_out[:, inds_sorted]
    W_rec_permuted = W_rec[:, inds_sorted]
    W_rec_permuted = W_rec_permuted[inds_sorted, :]

    N = W_rec_permuted.shape[0]
    n_inp = W_inp_permuted.shape[1]
    n_out = W_out_permuted.shape[0]

    fig, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [n_inp, N, n_out]})

    # Plotting the input matrix to the right
    ax[0].imshow(W_inp_permuted.T, aspect=1, cmap=cmap, vmin = -0.5, vmax=0.5)
    ax[0].set_title('Input Matrix', loc='center')
    ax[0].axis('off')
    # Plotting the recurrent matrix in the center
    ax[1].imshow(W_rec_permuted, aspect=1, cmap=cmap, vmin = -0.5, vmax=0.5)
    ax[1].set_title('Recurrent Matrix', loc='center')
    ax[1].axis('off')

    # # Plotting the input matrix to the right
    # ax[1, 1].imshow(input_matrix, aspect=1)
    # ax[1, 1].set_title('Input Matrix')
    # ax[1, 1].axis('off')

    # Plotting the output matrix underneath the recurrent matrix
    ax[2].imshow(W_out_permuted, aspect=1, cmap=cmap, vmin = -0.5, vmax=0.5)
    ax[2].set_title('Output Matrix', loc='center')
    ax[2].axis('off')

    # Hiding the empty subplot
    ax[0].axis('off')
    ax[1].axis('off')

    plt.tight_layout()
    if save:
        plt.savefig(path, dpi=300, transparent=True, bbox_inches='tight')
        path_png = path.split(".pdf")[0] + ".png"
        plt.savefig(path_png, dpi=300, transparent=True, bbox_inches='tight')
    if show:
        plt.show()
    return None

def plot_feature_array(Feature_array, labels_unraveled, show=True, save=False, path=None):
    colors = ["red", "green", "blue",
              "darkorange", "magenta", "cyan",
              "lime", "gray", "yellow",
              "deepskyblue", "olive", "purple",
              "pink", "brown", "salmon",
              "ghostwhite", "black", "lavender",
              "yellowgreen", "deeppink", "thistle"]

    n_clusters = len(np.unique(labels_unraveled))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    lbls = np.unique(labels_unraveled)
    for k, c in enumerate(lbls):
        inds = np.where(labels_unraveled == c)[0]
        ax.scatter(Feature_array[inds, 0], Feature_array[inds, 1], Feature_array[inds, 2],
                          edgecolor='b',
                          linewidth=0.1,
                          color=colors[k % len(colors)],
                          s=7,
                          alpha=1)
    plt.suptitle(f"Clustering of neural responses, relu RNN, 3D")
    if save:
        plt.savefig(path, dpi=300, transparent=True, bbox_inches='tight')
        path_png = path.split(".pdf")[0] + ".png"
        plt.savefig(path_png, dpi=300, transparent=True, bbox_inches='tight')
    if show:
        plt.show()
    return fig


def plot_RNN_distribution(label_list, show=True, save=False, path=None):
    # checking if the clusters come from the RNNs uniformly:
    labels_unraveled = list(chain.from_iterable(label_list))
    lbls = np.unique(labels_unraveled)
    n_RNNs = len(label_list)
    n_clusters = len(np.unique(lbls))
    Features_RNN = np.zeros((n_RNNs, n_clusters))

    for num_RNN in range(n_RNNs):
        labels = label_list[num_RNN]
        label_counts = np.array([np.sum(labels == i) for i in lbls])
        Features_RNN[num_RNN, :] = label_counts / np.sum(label_counts)

    pca = PCA(n_components=2)
    pca.fit(Features_RNN.T)
    points = pca.components_
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(points[0, :], points[1, :], color='magenta', edgecolors='k', linewidths=0.1)
    ax.set_title('PCA on the RNNs', fontsize=16)
    if save:
        plt.savefig(path, dpi=300, transparent=True, bbox_inches='tight')
        path_png = path.split(".pdf")[0] + ".png"
        plt.savefig(path_png, dpi=300, transparent=True, bbox_inches='tight')
    if show:
        plt.show()
    return fig



def plot_fixed_points(fixed_point_struct, fp_labels,
                      colors,
                      markers=None,
                      edgecolors=None,
                      n_dim=2, show=True, save=False, path=None):
    # PCA
    pca = PCA(n_components=n_dim)
    pca.fit(fixed_point_struct)
    data = fixed_point_struct @ pca.components_.T

    if n_dim == 2:
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    elif n_dim == 3:
        fig = plt.figure(figsize=(2, 2))
        ax = plt.axes(projection='3d')
        ax.view_init(45, -135)

    unique_labels = np.unique(fp_labels)
    for label in unique_labels:
        inds = np.where(np.array(fp_labels) == label)[0]
        k = int(label.split("_")[1])
        p = 1 if ("ufp" in label) else 0
        color = normalize_color(colors[k][p])
        marker = markers[k][p]
        edgecolor = 'k' if (edgecolors is None) else edgecolors[k][p]

        ax.scatter(*[data[inds, d] for d in range(n_dim)],
                   edgecolor=edgecolor,
                   marker=marker,
                   color=color,
                   linewidth=1,
                   alpha=0.7,
                   s=70)

    axes_names = ["x", "y", "z"][:n_dim]
    for a, axis in enumerate(axes_names):
        for postfix in ["min", "max"]:
            exec(f"{axis}_{postfix} = np.{postfix}(data[:, {a}])")
        exec(f"{axis}_range = 1.2 * ({axis}_max - {axis}_min)")
        exec(f"{axis}_mean = ({axis}_min + {axis}_max) / 2")
    for a, axis in enumerate(axes_names):
        exec(f"{axis}_min_new = ({axis}_mean - {axis}_range / 2)")
        exec(f"{axis}_max_new = ({axis}_mean + {axis}_range / 2)")
        exec(f"ax.set_{axis}lim([{axis}_min_new, {axis}_max_new])")
        exec(f"ax.set_{axis}ticklabels([])")
        exec(f"ax.set_{axis}ticks([])")
        exec(f"ax.set_{axis}ticklabels([])")

    if show:
        plt.show()
    if save:
        fig.savefig(path, dpi=300, bbox_inches='tight', transparent=True)
        path_png = path.split(".pdf")[0] + ".png"
        fig.savefig(path_png, dpi=300, transparent=True, bbox_inches='tight')
    plt.close(fig)
    return None


def plot_aligned_FPs(fp_list,
                     labels_list,
                     transforms_list,
                     colors_stable, colors_unstable, markers, edgecolors,
                     save=False, show=True, path=None, n_dim=2):

    unique_labels = np.unique(list(chain.from_iterable(labels_list)))
    fp_list_registered = [fp_list[i] @ transforms_list[i] for i in range(len(fp_list))]

    if n_dim == 2:
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    else:
        fig = plt.figure(figsize=(2, 2))
        ax = plt.axes(projection='3d')
        ax.view_init(45, -135)

    for i in range(len(fp_list)):
        for label in unique_labels:
            inds = np.where(np.array(labels_list[i]) == label)[0]

            color_ind = int(label.split("_")[1])
            marker = markers[color_ind][0] if "sfp" in label else markers[color_ind][1]
            edgecolor = normalize_color(list(edgecolors[color_ind][0] if "sfp" in label else edgecolors[color_ind][1]))
            color = colors_unstable[color_ind] if "ufp" in label else colors_stable[color_ind]
            ax.scatter(*(fp_list_registered[i][inds, k] for k in range(n_dim)),
                       marker=marker,
                       s=30,
                       color=color,
                       edgecolor=edgecolor)

    data = np.vstack(fp_list_registered)
    axes_names = ["x", "y"]
    if n_dim == 3:
        axes_names.append("z")
    for a, axis in enumerate(axes_names):
        for postfix in ["min", "max"]:
            exec(f"{axis}_{postfix} = np.{postfix}(data[:, {a}])")
        exec(f"{axis}_range = 1.2 * ({axis}_max - {axis}_min)")
        exec(f"{axis}_mean = ({axis}_min + {axis}_max) / 2")
        exec(f"{axis}_min_new = ({axis}_mean - {axis}_range / 2)")
        exec(f"{axis}_max_new = ({axis}_mean + {axis}_range / 2)")
        exec(f"ax.set_{axis}lim([{axis}_min_new, {axis}_max_new])")
        exec(f"ax.set_{axis}ticks([])")
        exec(f"ax.set_{axis}ticklabels([])")
    plt.tight_layout()
    if save:
        fig.savefig(path, dpi=300, transparent=True, bbox_inches='tight')
        path_png = path.split(".pdf")[0] + ".png"
        fig.savefig(path_png, dpi=300, transparent=True, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return None


def plot_psychometric_data(psychometric_data, show=True, save=False, path=None):
    coherence_lvls = psychometric_data["coherence_lvls"]
    # invert cause of the axes running from the bottom to the top
    Motion_rght_prcntg = psychometric_data["motion"]["right_choice_percentage"]#[::-1, :]
    Motion_MSE = psychometric_data["motion"]["MSE"]

    Color_rght_prcntg = psychometric_data["color"]["right_choice_percentage"]
    Color_MSE = psychometric_data["color"]["MSE"]#[::-1, :]
    num_lvls = Color_rght_prcntg.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(1 * 120 * mm, 60 * mm))

    for i, ctxt in enumerate(["Motion", "Color"]):
        im = axes[i].imshow(eval(f"{ctxt}_rght_prcntg"), cmap=cmap, interpolation="bicubic")

    ticks = ["-4", "-0.8", "0", "0.8", "4"]
    tick_inds = []
    for tick in ticks:
        for c, coh in enumerate(coherence_lvls):
            if np.abs(float(tick) - coh) < 0.0001:
                tick_inds.append(c)
                break

    axes[0].set_yticks(tick_inds)
    axes[0].set_yticklabels(ticks)
    axes[1].set_yticks([])
    for i in range(2):
        axes[i].set_xticks(tick_inds)
        axes[i].set_xticklabels(ticks, rotation=90)

    if show:
        plt.show()
    if save:
        fig.savefig(path, dpi=300, transparent=True, bbox_inches='tight')
        path_png = path.split(".pdf")[0] + ".png"
        fig.savefig(path_png, dpi=300, transparent=True, bbox_inches='tight')

    plt.close(fig)
    return None


def interpolate_color(low_color, mid_color, high_color,
                      low_val, mid_val, high_val,
                      val):
    if val < mid_val:
        coeffs = [(mid_val - val) / (mid_val - low_val),
                  (val - low_val) / (mid_val - low_val)]
        color = coeffs[0] * low_color + coeffs[1] * mid_color
    elif val > mid_val:
        coeffs = [(high_val - val) / (high_val - mid_val),
                  (val - mid_val) / (high_val - mid_val)]
        color = coeffs[0] * mid_color + coeffs[1] * high_color
    else:
        color = mid_color
    return color

import numpy as np
from matplotlib import pyplot as plt
from itertools import chain

def get_intercluster_connectivity(W_inps, W_recs, W_outs, labels_list):
    labels_unraveled = list(chain.from_iterable(labels_list))
    lbls = np.unique(labels_unraveled)

    num_inputs = W_inps[0].shape[1]
    num_outputs = W_outs[0].shape[0]

    input_to_cluster_dict = {}
    n_clusters = len(lbls)
    cluster_to_cluster_dict = {}
    cluster_to_output_dict = {}
    labels_by_RNN = []

    for i in range(num_inputs):
        for j in lbls:
            input_to_cluster_dict[f"({j},{i})"] = []

    for i in lbls:
        for j in range(num_outputs):
            cluster_to_output_dict[f"({j},{i})"] = []

    for i in lbls:
        for j in lbls:
            cluster_to_cluster_dict[f"({j},{i})"] = []

    # for each RNN in the list:
    for num_RNN in range(len(W_recs)):
        labels = labels_list[num_RNN]
        W_inp = W_inps[num_RNN]
        W_rec = W_recs[num_RNN]
        W_out = W_outs[num_RNN]
        for nrn in range(len(labels)):
            for inp_ch in range(W_inp.shape[1]):
                input_to_cluster_dict[f"({labels[nrn]},{inp_ch})"].append(W_inp[nrn, inp_ch])
            for out_ch in range(W_out.shape[0]):
                cluster_to_output_dict[f"({out_ch},{labels[nrn]})"].append(W_out[out_ch, nrn])
            for nrn_receiver in range(W_rec.shape[0]):
                cluster_to_cluster_dict[f"({labels[nrn_receiver]},{labels[nrn]})"].append(W_rec[nrn_receiver, nrn])
    ic_W_inp = np.zeros((n_clusters, num_inputs))
    ic_W_rec = np.zeros((n_clusters, n_clusters))
    ic_W_out = np.zeros((num_outputs, n_clusters))

    for i in lbls:
        for j in range(num_inputs):
            ic_W_inp[i, j] = np.mean(input_to_cluster_dict[f"({i},{j})"])

    for i in lbls:
        for j in lbls:
            ic_W_rec[i, j] = np.mean(cluster_to_cluster_dict[f"({i},{j})"])

    for i in range(num_outputs):
        for j in lbls:
            ic_W_out[i, j] = np.mean(cluster_to_output_dict[f"({i},{j})"])

    effective_connectivity_dict = {"ic_W_inp": ic_W_inp,
                                   "ic_W_rec": ic_W_rec,
                                   "ic_W_out": ic_W_out}


    ic_W_inp = effective_connectivity_dict["ic_W_inp"]
    ic_W_rec = effective_connectivity_dict["ic_W_rec"]
    ic_W_out = effective_connectivity_dict["ic_W_out"]
    return ic_W_inp, ic_W_rec, ic_W_out

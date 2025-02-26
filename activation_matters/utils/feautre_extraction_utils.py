import pickle
import numpy as np
from copy import deepcopy

from sklearn.decomposition import PCA
from itertools import chain


def get_dataset(df, filters):
    for attribute in list(filters.keys()):
        operand, value = filters[attribute]
        if type(value) == str:
            value = f'\'{value}\''
        filter_inds = eval(f"df.{attribute} {operand} {value}")
        if len(filter_inds) == 0:
            raise ValueError("Empty dataset!")
        df = df[filter_inds]
    df.sort_values(by=['RNN_score'], ascending=True)
    return df

def mean_downsample(array, window):
    k = array.shape[1]
    n_chunks = int(k // window)
    downsampled_array = np.hstack([np.mean(array[:, i*window: np.minimum(k, (i+1) * window)], axis = 1).reshape(-1, 1) for i in range(n_chunks)])
    return downsampled_array

def mean_after_integration(array, condition):
    InputDuration = condition["InputDuration"]
    features = np.mean(array[:, 10 + InputDuration :], axis = 1).reshape(-1, 1)
    return features

def means_over_epochs(array, times):
    k = array.shape[1]
    downsampled_array = np.hstack([np.mean(array[:, times[i]: times[i + 1]], axis=1).reshape(-1, 1) for i in  range(len(times) - 1)])
    return downsampled_array

def combine_data(Feature_array_list):
    K = Feature_array_list[0].shape[1]
    Feature_array = np.vstack([F for F in Feature_array_list]).reshape(-1, K)
    return Feature_array


def zscore_data(Feature_array, Feature_list):
    M = np.mean(Feature_array, axis=0).reshape(1, -1)
    STD = np.std(Feature_array, axis=0)
    STD[np.where(STD == 0)[0]] = 1
    STD = STD.reshape(1, -1)

    N = Feature_array.shape[0]
    Feature_array = (Feature_array - np.repeat(M, repeats=N, axis=0)) / np.repeat(STD, repeats=N, axis=0)
    Feature_list = [(F - np.repeat(M, repeats=F.shape[0], axis=0)) / np.repeat(STD, repeats=F.shape[0], axis=0) for F in Feature_list]
    return Feature_array, Feature_list

def normalize_data(Feature_array, Feature_list):
    normalization_coeffs = np.linalg.norm(Feature_array, axis=1)
    normalization_coeffs_inds = np.where(normalization_coeffs == 0)[0]
    # if the norm is equal to zero, leave it unaffected
    normalization_coeffs[normalization_coeffs_inds] = 1.0
    Feature_array = (Feature_array.T / normalization_coeffs).T

    Feature_array_list_normalized = []
    for F in Feature_list:
        c = np.linalg.norm(F, axis=1)
        c[np.where(c == 0)[0]] = 1.0
        Feature_array_list_normalized.append((F.T / c).T)
    Feature_list = Feature_array_list_normalized
    return Feature_array, Feature_list

def reduce_dimensionality(Feature_array, Feature_list, var_thr=0.99):
    pca = PCA(n_components=np.minimum(100, Feature_array.shape[1]))
    pca.fit(Feature_array)
    try:
        n_dim = np.maximum(np.where(np.cumsum(pca.explained_variance_ratio_) > var_thr)[0][0], 3)
    except:
        n_dim = np.maximum(np.where(np.cumsum(pca.explained_variance_ratio_) > var_thr)[0][0], 3)
    print(f"Number of dimensions after PCA: {n_dim}")
    Pr = pca.components_.T[:, :n_dim]
    Feature_array = Feature_array @ Pr

    # need to further normalize it by the std of the first dimension!
    sigma = np.std(Feature_array[:, 0])
    Feature_array = Feature_array / sigma
    Feature_list = [(F @ Pr) / sigma for F in Feature_list]
    return Feature_array, Feature_list

def remove_low_contributing_neurons(W_inp, W_rec, W_out, RNN_trajectories, trajectory_features, quantile=0.3):

    mfr = np.mean(np.mean(np.abs(RNN_trajectories), axis=1), axis=1)
    mfr_thr = np.quantile(mfr, quantile)

    inds = np.where((mfr > mfr_thr))[0]

    W_rec_filtered = W_rec[inds, :]
    W_rec_filtered = W_rec_filtered[:, inds]
    W_inp_filtered = deepcopy(W_inp[inds, :])
    W_out_filtered = deepcopy(W_out[:, inds])

    RNN_trajectories_filtered = RNN_trajectories[inds, ...]
    trajectory_features_filtered = trajectory_features[inds, ...]
    return W_inp_filtered, W_rec_filtered, W_out_filtered, RNN_trajectories_filtered, trajectory_features_filtered

def extract_features(dataframe,
                     processing_params,
                     downsample_window=10,
                     Dales_principle=True,
                     connectivity_importance_factor=5,
                     nrn_type_importance_factor=10):

    RNN_trajectories_list = dataframe["RNN_trajectories"].tolist()
    W_inp_list = dataframe["W_inp_RNN"].tolist()
    W_rec_list = dataframe["W_rec_RNN"].tolist()
    W_out_list = dataframe["W_out_RNN"].tolist()
    scores = dataframe["RNN_score"].tolist()

    W_inp_filtered_list = []
    W_rec_filtered_list = []
    W_out_filtered_list = []

    trajectories_filtered_list = []
    trajectory_features_filtered_list = []
    input_features_filtered_list = []
    recurrent_features_outgoing_filtered_list = []
    output_features_filtered_list = []

    if Dales_principle:
        nrn_types_list = []

    num_RNNs = len(RNN_trajectories_list)
    TrajectoryFeature_array_list = []
    Connectivity_feature_array_list = []

    for i in range(num_RNNs):
        trajectories = RNN_trajectories_list[i]
        K = trajectories.shape[-1]
        trajectory_features = []
        for k in range(K):
            trajectory_features.append(mean_downsample(trajectories[:, :, k], window=downsample_window))
        trajectory_features = np.hstack(trajectory_features)

        W_inp = W_inp_list[i]
        W_rec = W_rec_list[i]
        W_out = W_out_list[i]

        # remove low contributing neurons and neurons with low variance!
        W_inp_filtered, W_rec_filtered, W_out_filtered, trajectories_filtered, trajectory_features_filtered =\
            remove_low_contributing_neurons(W_inp, W_rec, W_out, trajectories, trajectory_features)

        W_inp_filtered_list.append(W_inp_filtered)
        W_rec_filtered_list.append(W_rec_filtered)
        W_out_filtered_list.append(W_out_filtered)

        trajectories_filtered_list.append(trajectories_filtered)
        trajectory_features_filtered_list.append(trajectory_features_filtered)

        input_features_filtered_list.append(deepcopy(W_inp_filtered))
        output_features_filtered_list.append(deepcopy(W_out_filtered.T))

        #####
        res_dwnstrm = W_rec_filtered @ trajectory_features_filtered
        recurrent_features_outgoing_filtered_list.append(res_dwnstrm)
        #####

        if Dales_principle == True:
            inds_inh = np.where(np.sum(W_rec_filtered, axis=0) < 0)
            neuron_types_in_RNN = np.ones(W_rec_filtered.shape[0])
            neuron_types_in_RNN[inds_inh] = -1
            nrn_types_list.append((neuron_types_in_RNN.reshape(-1, 1)))

    Trajectory_features = combine_data(trajectory_features_filtered_list)
    Trajectory_features_list = trajectory_features_filtered_list

    if ("Trajectory" in processing_params.keys()):
        if processing_params["Trajectory"]["zscore"]:
            Trajectory_features, Trajectory_features_list = zscore_data(Trajectory_features,
                                                                        Trajectory_features_list)
        if processing_params["Trajectory"]["normalize"]:
            Trajectory_features, Trajectory_features_list = normalize_data(Trajectory_features,
                                                                           Trajectory_features_list)
        if processing_params["Trajectory"]["do_pca"]:
            var_thr = processing_params["Trajectory"]["var_thr"]
            Trajectory_features, Trajectory_features_list = reduce_dimensionality(Trajectory_features,
                                                                                  Trajectory_features_list,
                                                                                  var_thr=var_thr)
    # recurrent_features_list = [W_rec_filtered_list[i] @ tr for i, tr in enumerate(Trajectory_features_list)]
    # Input_features = combine_data(input_features_filtered_list)
    # Recurrent_features = combine_data(recurrent_features_list)
    # Output_features = combine_data(output_features_filtered_list)
    # Connectivity_features = np.hstack([Input_features, Output_features, Recurrent_features])
    # Connectivity_features_list = [np.hstack([inp_f, out_f, rec_f]) for inp_f, out_f, rec_f in
    #                               zip(input_features_filtered_list,
    #                                   output_features_filtered_list,
    #                                   recurrent_features_list)]

    Input_features = combine_data(input_features_filtered_list)
    Output_features = combine_data(output_features_filtered_list)
    Connectivity_features = np.hstack([Input_features, Output_features])
    Connectivity_features_list = [np.hstack([inp_f, out_f]) for inp_f, out_f in
                                  zip(input_features_filtered_list,
                                      output_features_filtered_list)]

    if ("Connectivity" in processing_params.keys()):
        if processing_params["Connectivity"]["zscore"]:
            Connectivity_features, Connectivity_features_list = zscore_data(Connectivity_features, Connectivity_features_list)
        if processing_params["Connectivity"]["normalize"]:
            Connectivity_features, Connectivity_features_list = normalize_data(Connectivity_features, Connectivity_features_list)
        if processing_params["Connectivity"]["do_pca"]:
            var_thr = processing_params["Connectivity"]["var_thr"]
            Connectivity_features, Connectivity_features_list = reduce_dimensionality(Connectivity_features, Connectivity_features_list, var_thr=var_thr)

    if ("Connectivity" in processing_params.keys()) and ("Trajectory" in processing_params.keys()):
        ImpF = connectivity_importance_factor
        Feature_array_combined = np.hstack([Trajectory_features, ImpF * Connectivity_features])
        Feature_list_combined = [np.hstack([tr_f, ImpF * conn_f]) for tr_f, conn_f in zip(Trajectory_features_list, Connectivity_features_list)]
    elif ("Connectivity" in processing_params.keys()) and not ("Trajectory" in processing_params.keys()):
        Feature_array_combined = Connectivity_features
        Feature_list_combined = Connectivity_feature_array_list
    elif not ("Connectivity" in processing_params.keys()) and ("Trajectory" in processing_params.keys()):
        Feature_array_combined = Trajectory_features
        Feature_list_combined = TrajectoryFeature_array_list

    if Dales_principle:
        nrn_types_combined = np.array(list(chain.from_iterable(nrn_types_list)))
        ImpF = nrn_type_importance_factor
        Feature_array_combined = np.hstack([Feature_array_combined, ImpF * nrn_types_combined.reshape(-1, 1)])
        Feature_list_combined = [np.hstack([f, ImpF * t.reshape(-1, 1)]) for f, t in zip(Feature_list_combined, nrn_types_list)]

    data = {"Features": Feature_array_combined,
            "Features_by_RNN": Feature_list_combined,
            "Trajectories_by_RNN": trajectories_filtered_list,
            "Trajectory_Features": Trajectory_features,
            "Trajectory_Features_by_RNN": TrajectoryFeature_array_list,
            "W_inp_list": W_inp_filtered_list,
            "W_rec_list": W_rec_filtered_list,
            "W_out_list": W_out_filtered_list,
            "RNN_score": scores}

    if Dales_principle:
        data["nrn_type"] = nrn_types_combined
        data["nrn_type_list"] = nrn_types_list
    return data


def extract_features_single_RNN(RNN, task, mask,
                                processing_params,
                                downsample_window=10,
                                Dales_principle=True,
                                connectivity_importance_factor=5,
                                nrn_type_importance_factor=10):
    input_batch, target_batch, conditions = task.get_batch()
    RNN.clear_history()
    RNN.run(input_timeseries=input_batch,
                 sigma_rec=0,
                 sigma_inp=0)

    trajectories = RNN.get_history()
    K = trajectories.shape[-1]
    trajectory_features = []
    for k in range(K):
        trajectory_features.append(mean_downsample(trajectories[:, :, k], window=downsample_window))
    trajectory_features = np.hstack(trajectory_features)

    mse = lambda x, y: np.sum((x - y) ** 2)
    # nrn_scores = score_neurons(RNN, task, scoring_function=mse, mask=mask)
    # # kneedle = KneeLocator(np.arange(len(nrn_scores)), sorted(nrn_scores), S=0.5,
    # #                       curve="convex",
    # #                       direction="increasing")
    # # knee = int(kneedle.knee)
    # # inds = np.where(nrn_scores >= sorted(nrn_scores)[knee])[0]
    # # if len(inds) < 20: # always have at least 20 neurons
    # #     inds = np.where(nrn_scores >= sorted(nrn_scores)[-20])[0]
    #
    # inds = np.where(nrn_scores >= 0.02)[0]
    # if len(inds) < 20:
    #     knee = len(nrn_scores) - 20  # always have at least 20 neurons
    #     inds = np.where(nrn_scores >= sorted(nrn_scores)[knee])[0]
    inds = np.arange(RNN.W_rec.shape[0])
    W_rec_filtered = RNN.W_rec[inds, :]
    W_rec_filtered = W_rec_filtered[:, inds]
    W_inp_filtered = deepcopy(RNN.W_inp[inds, :])
    W_out_filtered = deepcopy(RNN.W_out[:, inds])

    trajectory_features = trajectory_features[inds, ...]

    if Dales_principle == True:
        inds_inh = np.where(np.sum(W_rec_filtered, axis=0) < 0)
        nrn_types = np.ones(W_rec_filtered.shape[0])
        nrn_types[inds_inh] = -1

    if ("Trajectory" in processing_params.keys()):
        if processing_params["Trajectory"]["zscore"]:
            M = np.nanmean(trajectory_features, axis=0).reshape(1, -1)
            STD = np.nanstd(trajectory_features, axis=0).reshape(1, -1)
            STD[np.where(STD == 0)[0]] = 1
            N = trajectory_features.shape[0]
            trajectory_features = (trajectory_features - np.repeat(M, repeats=N, axis=0)) / np.repeat(STD, repeats=N, axis=0)

        if processing_params["Trajectory"]["normalize"]:
            M = np.nanmean(trajectory_features, axis = 1).reshape(-1, 1)
            STD = np.nanstd(trajectory_features, axis=1).reshape(-1, 1)
            STD[np.where(STD == 0)[0]] = 1
            K = trajectory_features.shape[1]
            trajectory_features = (trajectory_features - np.repeat(M, repeats=K, axis=1)) / np.repeat(STD, repeats=K, axis=1)

        if processing_params["Trajectory"]["do_pca"]:
            var_thr = processing_params["Trajectory"]["var_thr"]
            pca = PCA(n_components=np.minimum(trajectory_features.shape[0], trajectory_features.shape[1]))
            pca.fit(trajectory_features)
            n_dim = np.maximum(np.where(np.cumsum(pca.explained_variance_ratio_) > var_thr)[0][0], 3)
            print(f"Number of dimensions after PCA: {n_dim}")
            Pr = pca.components_.T[:, :n_dim]
            trajectory_features = trajectory_features @ Pr
            trajectory_features = trajectory_features - np.mean(trajectory_features, axis=0)
            STD = np.std(trajectory_features[:, 0])
            trajectory_features = trajectory_features/ STD


        input_features = W_inp_filtered
        output_features = W_out_filtered.T

        connectivity_features = np.hstack([input_features, output_features])

    if ("Connectivity" in processing_params.keys()):
        if processing_params["Connectivity"]["zscore"]:
            M = np.mean(connectivity_features, axis = 0).reshape(1, -1)
            STD = np.std(connectivity_features, axis = 0).reshape(1, -1)
            STD[np.where(STD == 0)[0]] = 1
            N = connectivity_features.shape[0]
            connectivity_features = (connectivity_features - np.repeat(M, repeats=N, axis=0)) / np.repeat(STD, repeats=N, axis=0)
        if processing_params["Connectivity"]["normalize"]:
            M = np.mean(connectivity_features, axis=1).reshape(-1, 1)
            STD = np.std(connectivity_features, axis=1).reshape(-1, 1)
            STD[np.where(STD == 0)[0]] = 1
            K = connectivity_features.shape[1]
            connectivity_features = (connectivity_features - np.repeat(M, repeats=K, axis=1)) / np.repeat(STD, repeats=K, axis=1)
        if processing_params["Connectivity"]["do_pca"]:
            var_thr = processing_params["Connectivity"]["var_thr"]
            pca = PCA(n_components=np.min([100, connectivity_features.shape[1], connectivity_features.shape[0]]))
            pca.fit(connectivity_features)
            n_dim = np.maximum(np.where(np.cumsum(pca.explained_variance_ratio_) > var_thr)[0][0], 3)
            print(f"Number of dimensions after PCA: {n_dim}")
            Pr = pca.components_.T[:, :n_dim]
            connectivity_features = connectivity_features @ Pr
            # Z score the data afterward
            connectivity_features = connectivity_features - np.mean(connectivity_features, axis=0)
            STD = np.std(connectivity_features[:, 0])
            connectivity_features = connectivity_features/ STD

    if ("Connectivity" in processing_params.keys()) and ("Trajectory" in processing_params.keys()):
        ImpF = connectivity_importance_factor
        Feature_array_combined = np.hstack([trajectory_features, ImpF * connectivity_features])
    elif ("Connectivity" in processing_params.keys()) and not ("Trajectory" in processing_params.keys()):
        Feature_array_combined = connectivity_features
    elif not ("Connectivity" in processing_params.keys()) and ("Trajectory" in processing_params.keys()):
        Feature_array_combined = trajectory_features

    if Dales_principle:
        ImpF = nrn_type_importance_factor
        Feature_array_combined = np.hstack([Feature_array_combined, ImpF * nrn_types.reshape(-1, 1)])

    data = {"Features": Feature_array_combined,
            "Trajectory_Features": trajectory_features,
            "W_rec" : W_rec_filtered,
            "W_inp": W_inp_filtered,
            "W_out": W_out_filtered,
            "Connectivity_Features": connectivity_features}
    if Dales_principle:
        data["nrn_type"] = nrn_types
    return data
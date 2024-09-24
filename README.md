## Analysing dynamics, trajectories, neural tuning, and behavior of continuous-time RNNs with different architectural choices

This repository contains the code for running analysis on how the seemingly inconsequential RNN design-choices affect the dynamics, trajectories, neural tuning and, finally, behavior.

### Setting up config files

You would require to set the paths in the `congigs/task` files.
Make sure to modify the following, setting *absolute* paths:

- dataset_path (contains the top 30 RNNs, located in `../data/RNN_datasets`)
- auxilliary_datasets_path (`../data/auxilliary_datasets)`
- img_folder (`../img/task name)`
- fixed_points_data_folder (`../fixed_points/task name)`

### Dependencies

This package relies on a [`trainRNNbrain`](link) package, make sure to download it and install it locally by running `python -m pip install -e ./` from the `trainRNNbrain` folder.

### How to run the analysis

The dataset containing top 30 networks for each of the architectural class are stored in `data/RNN_datasets`.

- To analyze the (dis)similarity of RNNs in terms of **trajectories**, run `1_trajectory_similarity_analysis.py`.
The code generates the trajecotries from the networks (as well as the networks with shuffled connectivity, used as control), projects them with PCA, puts all trajectories to the same scale. Further, it computes the pairwise distances between the sets of trajectories and then produces MDS embedding. You would need to plot the trajectories and MDS embedding with `plots/plotting_trajectories.py`

- To analyze the (dis)similarity of RNNs in terms of their **single-unit selectivity** configurations, run `2_single_unit_selectivity_similarity_analysis.py`.
The code generates the trajecotries from the networks (as well as the networks with shuffled connectivity, used as control), reduces the dimensionality with PCA down to N x n_PCs (N - number of neurons in the RNN), puts all configurations to the same scale. It then computes the pairwise distances between the sets of single-unit selectivity and produces MDS embedding.
To plot single-unit selectivity, run `plots/plotting_single_unit_selectivity.py`

- For the differences in RNNs in terms of their **fixed point configurations** run `3.1_computing_fp.py` first (it computes the fixed points, and it takes awhile), followed by running `3.2_fp_similarity_analysis.py`, which then compares the fixed point configurations and plots MDS embedding.
To plot fixed point analysis, run `plots/plotting_fixed_point_configurations.py`

- To analyze **trial endpoint configurations**, run `4_trajectory_endpoints_similarity_analysis.py`, and then plot it with `plotting_trajectory_endpoints.py`

- Finally, to analyze the discrepancies in the behavior of RNNs with different activations functions on the context-dependent decision-making task, run `5_behavior_analysis_CDDM.py`.

As a result the code produces multiple figures (trajectories, single-unit selectivity, fixed point configurations, and trial end point configurations for each RNN, and MDS embeddings for each aspect) in the corresponding `img/taskname` folder.






 





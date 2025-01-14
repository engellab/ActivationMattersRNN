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
This package relies on a [`trainRNNbrain`](https://github.com/engellab/trainRNNbrain) and [`latent_circiut_inference`](https://github.com/engellab/latent_circuit_inference) packages, make sure to download it and install it locally by running `python -m pip install -e ./` from the `trainRNNbrain`  and `latent_circuit_inference' folder.

| Package Name               | Version     |
|----------------------------|-------------|
| **ActivationMatters**      | **1.0**     |
| antlr4-python3-runtime     | 4.9.3       |
| contourpy                  | 1.3.1       |
| cycler                     | 0.12.1      |
| filelock                   | 3.16.1      |
| fonttools                  | 4.55.3      |
| fsspec                     | 2024.12.0   |
| hydra-core                 | 1.3.2       |
| Jinja2                     | 3.1.5       |
| joblib                     | 1.4.2       |
| kiwisolver                 | 1.4.8       |
| **latent_circuit_inference** | **0.1**    |
| MarkupSafe                 | 3.0.2       |
| matplotlib                 | 3.10.0      |
| mpmath                     | 1.3.0       |
| networkx                   | 3.4.2       |
| numdifftools               | 0.9.41      |
| numpy                      | 2.2.1       |
| omegaconf                  | 2.3.0       |
| packaging                  | 24.2        |
| pandas                     | 2.2.3       |
| pillow                     | 11.1.0      |
| pip                        | 24.2        |
| pyparsing                  | 3.2.1       |
| python-dateutil            | 2.9.0.post0 |
| pytz                       | 2024.2      |
| PyYAML                     | 6.0.2       |
| scikit-learn               | 1.6.1       |
| scipy                      | 1.15.1      |
| setuptools                 | 75.1.0      |
| six                        | 1.17.0      |
| sympy                      | 1.13.1      |
| threadpoolctl              | 3.5.0       |
| torch                      | 2.5.1       |
| tqdm                       | 4.67.1      |
| **trainRNNbrain**          | **1.0**     |
| typing_extensions          | 4.12.2      |
| tzdata                     | 2024.2      |
| wheel                      | 0.44.0      |


### How to run the analysis
Modify the config variables in config/task folders, so that the path-variables point towards the correct folders.

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






 





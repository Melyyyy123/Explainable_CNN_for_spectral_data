# Workflow of building the model for the derivation of the subsample of the dps1200 data

## Getting to know the data
In order to get to know the data, some plots and descriptive statistics were generated which can be seen in dps1200Deriv_sub.ipynb.
To train the model the vector normalized, smoothed and derivated (2nd derivation) (Savitzky–Golay algorithm) data was taken (https://www.nature.com/articles/s41598-020-68194-w).
Additionally a feature selection step was conducted which is expected to lead to better results (https://arxiv.org/pdf/2005.07530.pdf).

## The model
The optimal model and hyperparameters of sub_data (the subsample of the dps1200 data) was taken, optimized_model.ipynb. 

## Comments
The sub_architecture did not perform well with the sub_deriv_data.
The reason for this could be the fact that the second derivation comes with a loss of information. Therefore it would be necessary to optimize a new architecture.
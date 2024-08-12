# Workflow of Building the Model for the dps1200 data (all_data)

## 01 Getting to know the data
This dataset contains the whole infrared spectra from 4000-400 cm⁻¹ of the pine wood samples.
In order to get to know the data, some plots and descriptive statistics were generated which can be seen in 01_dps1200_all.ipynb.
To train the model the vector normalized and smoothed (Savitzky–Golay algorithm) data was taken (https://www.nature.com/articles/s41598-020-68194-w).

## 02 First try with the LeNet
The first model used was the LeNet model with the same hyperparameters used for architecture optimization of the sub_data, 02_LeNet.ipynb. The initial results were promising so I chose to optimize the LeNet architecture.

## 03 Sample weighing
As the densitiy in old samples is lower than in younger samples, a function which calculates the sample weights was added. 

## 03 Cross validation
Since CNN training involves random sampling and weights initialization (in this case), it is usefull to use cross-validation.

## 03 Hyperparameter Tuning
In order to find the optimal hyperparameter, a Bayesian Optimization Section was implemented, Bayesoptimization.py. This process was extremely computational intensive and it was necessary to rent a server from Lambdalab. The server had a gpu_1x_a10. The optimal model parameters can be found in output.txt and the plots of monitoring the training process can be found in convergence_plot.png.

## 04 Optimized LeNet Stability test
The optimal architecture was tested on performance with different random seeds in order to detect the influence of randomness (random weight initialization) on the model, 04_optimized_lenet.ipynb.

## 05 Save the optimal model
The optimal model was saved, so that it can be reloaded several times. Script: 05_final_model_all.ipynb and saved in dps1200all_model.keras.

# Workflow on explaining the prediction of the model

## 06 Occlusion sensitivity
Occlusion sensitivity is a perturbation based method. 
The importance of the input data was tested with the occlusion sensitivity. Script: 06_occlusion_all.ipynb.

## 07 Guided backpropagation
The guided backpropagation is a gradient-based visualization technique. The code and the output can be seen in 07_guided_back_all.ipynb. 

## 08 Layer-wise relevance propagation
The method LRP is based on propagating the predictions backward through the neural network, using a set of purposely designed propagation rules. The code and the output can be seen in 08_LRP_all.ipynb.

## 09 Shap Additive Explanations
Shapley values are a widely used approach from cooperative game theory that come with desirable properties. The code and the output can be seen in 09_shap_all.ipynb.
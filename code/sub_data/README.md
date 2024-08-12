# Workflow of building the model for sub_data (subsample of the dps1200 data)

## 01 Getting to know the data
This dataset contains only a selection of bandwidths of the all_data. These bandwidths are selected with the help of PLS Regression. Further information: https://www.nature.com/articles/s41598-020-68194-w.

In order to get to know the data, some plots and descriptive statistics were generated which can be seen in 01_dps1200_sub.ipynb.
To train the model, the vector normalized and smoothed (Savitzky–Golay algorithm) data was taken (https://www.nature.com/articles/s41598-020-68194-w).
The feature selection step is expected to lead to better results (https://arxiv.org/pdf/2005.07530.pdf).

## 02 First try: Bjerrum et al.
The first architecture which was tested, was defined by Bjerrum et al. The code and the parameters of https://github.com/dario-passos/DeepLearning_for_VIS-NIR_Spectra/blob/master/notebooks/Bjerrum2017_CNN/BayesOpt_CNN1.2.ipynb were used to obtain an initial model.

## 03 Second try: LeNet
The second architecture which was tested, was defined by Yann LeCun. The code and the parameters of https://github.com/dario-passos/DeepLearning_for_VIS-NIR_Spectra/blob/master/notebooks/Bjerrum2017_CNN/BayesOpt_CNN1.2.ipynb were used to obtain an initial model.

## 04 Third try with: Tegegn et al.
The third architecture which was tested, was defined by Tegegn et al. in the paper "Convolutional Neural Networks for Quantitative Prediction of Different
Organic Materials using Near-Infrared Spectrum".

## Obtain the architecture
In the next step the model architecture was build from scratch to obtain an architecture optimized on the sub_data. The start was a samll architecture with one convolutional layer, one flatten layer and one dense layer. The RMSE of the test set stopped increasing with 5 dense Layers. In the next step 1 dropout layer was added. Then, the number of convolutional layers for 4 and 5 dense layers were tested. After 3 convolutional layers the test RMSE did not improve any more. Over all, 4 dense layers performed better with more convolutional layers than 5 dense layers. Then, different possibilities of pooling layer arrangements were tested. The best result achieved 1 Pooling layer before the flatten layer.

The hyperparameters used were initial guesses inspired by Tegegn et al and Bjerrum et al.
As loss function the Huber loss was used as suggested by Bjerrum et al.
The optimizer used is 'Adam' which was recommended by Mr. Passos who rebuild the Bjerrum Model. 

The different tests of the architectures can be found in the folder raw_tuning.
The plots for monitoring the RMSE and the R² of the different architectures can be found in architecture_tuning.ipynb.
The plot for comparing the convolutional layers with 4 and 5 dense layers can also be found in architecture_tuning.ipynb.

Attention: These tests are not reproducible because no random seed was set. If the random weight initialization assigns important features a small weight, the model will yield in poor predictions. In order to explore the behavior of different architectures without bias, no random seed was set.

## Sample weighing
As the data densitiy in old samples is lower than in younger samples, a function which calculates the sample weights was added. 

## Cross validation
Since CNN training involves random sampling and weights initialization (in this case), it is usefull to use cross-validation.

## 05 Hyperparameter Tuning
In order to find the optimal hyperparameter, a Bayesian Optimization Section was implemented, 05_hyperparametertuning.ipynb. This process was extremely computational intensive.

## 06 Stability test
The optimal architecture was tested on performance with different random seeds in order to detect the influence of random weight initialization on the model, 
06_stabilitytest.ipynb.

## 07 Save the optimal model
The optimal model was saved so that it can be reloaded several times. Script: 07_model_final_sub.ipynb and saved in dps1200sub_model.keras.

# Workflow on explaining the prediction of the model

## 08 Occlusion sensitivity
The occlusion sensitivity is a perturbation based method.
The importance of the input data was tested with the occlusion sensitivity. Script: 08_occlusion_sub.ipynb.

## 09 Guided backpropagation
The guided backpropagation is a gradient-based visualization technique. The code and the output can be seen in 09_guided_back_sub.ipynb. 

## 10 Layer-wise relevance propagation
The method LRP is based on propagating the predictions backward through the neural network, using a set of purposely designed propagation rules. The code and the output can be seen in 10_LRP_sub.ipynb.

## 11 Shap Additive Explanations
Shapley values are a widely used approach from cooperative game theory that come with desirable properties. The code and the output can be seen in 11_shap_sub.ipynb.
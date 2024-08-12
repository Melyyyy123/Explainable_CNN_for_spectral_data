#!/usr/env python3

# Bayesian Optimization of the LeNet architecture

# Warning: This script is very computationally intensive

# import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score

import keras

import tensorflow as tf
from keras.callbacks import EarlyStopping,ReduceLROnPlateau


# seeds
def set_reproducible():
    np.random.seed(12345)
    random.seed(12345)
    tf.random.set_seed(12345)
    
set_reproducible()

#script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
#rel_path = "../dps1200_all.csv"
#abs_file_path = os.path.join(script_dir, rel_path)

dps1200 = pd.read_csv("dps1200_all.csv")
features = dps1200.iloc[:, 4:].values
labels = dps1200.iloc[:, 0].values

# Calculate Sample weights

def convertToDecade(y:int) -> int: 
    return int(str(y)[:3])

def calculate_sample_weights(y_train):

    decades = [convertToDecade(year) for year in y_train]

    unique_decades, counts = np.unique(decades, return_counts=True)
    total_samples = len(y_train)
    
    weights = {}
    for decade, count in zip(unique_decades, counts):
        weights[decade] = 1 - count/total_samples

    sample_weights = []
    for year in y_train:
        sample_weights.append(weights[convertToDecade(year)])
        
    return np.array(sample_weights)

# Model Architecture

def LeNet_model(DenseN, DropoutR, C1_K, C1_S, C2_K, C2_S, input_dim):
    
    model = keras.Sequential()
    model.add(keras.layers.Input((input_dim, 1)))
    model.add(keras.layers.GaussianNoise(0.0001))

    model.add(keras.layers.Conv1D(C1_K, (C1_S), padding='valid', activation='relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=2))

    model.add(keras.layers.Conv1D(C2_K, (C2_S), padding='valid', activation='relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(DropoutR))

    model.add(keras.layers.Dense(DenseN, activation='relu'))
    model.add(keras.layers.Dense(1, activation='relu'))
    

    model.compile(loss='huber_loss', optimizer=keras.optimizers.Adam(learning_rate=0.004), metrics=['mean_absolute_error'])
    
    return model

## Compute error metrics
def error_metrices(y_true_train, y_predicted_train, y_true_test, y_predicted_test, verbose=True):
    rmse_train = np.sqrt(mean_squared_error(y_true_train, y_predicted_train))
    rmse_test = np.sqrt(mean_squared_error(y_true_test, y_predicted_test))
    R2_train= r2_score(y_true_train, y_predicted_train)
    R2_test= r2_score(y_true_test, y_predicted_test)
    h = tf.keras.losses.Huber()
    hub_train = h(y_true_train, y_predicted_train).numpy()
    hub_test = h(y_true_test, y_predicted_test).numpy()

    if verbose:
        print('*********** Benchmark results ***********\n')
        print(f"R2    (Train/Test) = {R2_train:.3f} / {R2_test:.3f}")
        print(f"RMSE  (Train/Test) = {rmse_train:.3f} / {rmse_test:.3f}")
        print(f"Huber (Train/Test) = {hub_train:.3f} / {hub_test:.3f}")

    return (rmse_train, rmse_test, R2_train, R2_test, hub_train, hub_test)

class ModelWithData:
    def __init__(self, model, train_x, train_label, train_predicted, test_x, test_label, test_predicted, score:float, iteration) -> None:
        self.model = model
        self.train_x = train_x
        self.train_label = train_label
        self.train_predicted = train_predicted
        self.test_x = test_x
        self.test_label = test_label
        self.test_predicted = test_predicted
        self.score = score
        self.iteration = iteration

    def isBetter(self, otherScore: float) -> bool:
        return otherScore < 0 or (self.score >= 0 and self.score <= otherScore)
    


# X_train und y_train are features and labels
def evaluations_of_models(features, labels, DenseN, DropoutR, C1_K, C1_S, C2_K, C2_S, verbose=True):

    x = np.array(features)
    y = np.array(labels)
    input_dim = 1866
    epochs = 800
    
    # generate model
    model = LeNet_model(DenseN, DropoutR, C1_K, C1_S, C2_K, C2_S, input_dim)

    # Define the number of folds for Cross-Validation
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores_train = { "rmse": [], "r2": [], "huber": []}
    scores_test = { "rmse": [], "r2": [], "huber": []}

    verbose_param = 0

    if verbose:
        verbose_param = 1


    monitor = EarlyStopping(monitor='val_loss', min_delta=4e-5, patience=50, verbose=0, mode='auto', restore_best_weights=True)
    rdlr = ReduceLROnPlateau(patience=25, factor=0.5, min_lr=1e-6, monitor='val_loss', verbose=verbose_param)

    bestMwd = ModelWithData(None, 0, 0, 0, 0, 0, 0, -1, 0)
    
    # Iterate through the folds and train/test the model
    i = 0
    for train_index, test_index in skf.split(x, y):
        i = i + 1
        if verbose:
            print(f'\n\n> Iteration {i}')

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Training on X_train_fold, y_train_fold

        sample_weight=calculate_sample_weights(y_train)
        g = model.fit(x_train, y_train, epochs=epochs, batch_size=45, validation_data=(x_test, y_test), verbose=0, callbacks=[rdlr, monitor], sample_weight=sample_weight)

        # Evaluation of the model
        accuracy = model.evaluate(x_test, y_test, verbose=0)
        
        if verbose:
            print(f'Fold Accuracy: {accuracy}')

        # Analyse the error metrics

        train_pred = model.predict(x_train, verbose=0)
        test_pred = model.predict(x_test, verbose=0)

        (rmse_train, rmse_test, r2_train, r2_test, huber_train, huber_test) = error_metrices(y_train, train_pred, y_test, test_pred, verbose=verbose)

        mwd = ModelWithData(g, x_train, y_train, train_pred, x_test, y_test, test_pred, rmse_train, i)

        if mwd.isBetter(bestMwd.score):
            bestMwd = mwd

        scores_train["rmse"].append(rmse_train)
        scores_train["r2"].append(r2_train)
        scores_train["huber"].append(huber_train)

        scores_test["rmse"].append(rmse_test)
        scores_test["r2"].append(r2_test)
        scores_test["huber"].append(huber_test)

    ## clear session 
    keras.backend.clear_session()

    scores_train_mean = {}
    scores_test_mean = {}
    num_model = len(scores_train["rmse"])

    for metric in scores_train.keys():
        scores_train_mean[metric] = np.mean(scores_train[metric])
        scores_test_mean[metric] = np.mean(scores_test[metric])
        if verbose:
            print(f'Train: {metric} (mean of {num_model} models)= {scores_train_mean[metric]} \nTest: {metric} (mean of {num_model} models)= {scores_test_mean[metric]}')

    return (bestMwd, scores_train_mean, scores_test_mean)

##### Optimization #######

# Libraries
import GPyOpt

## Range of parameters used by Bjerrum et al. 2017
dense_range=np.arange(4,720,1)
c1_k_range=np.arange(2,64,1)
c1_s_range=np.arange(1,128,1)
c2_k_range=np.arange(2,64,1)
c2_s_range=np.arange(1,128,1)

# bounds for hyperparameters in the model
bounds = [{'name': 'DenseN',   'type': 'discrete',  'domain': dense_range},
          {'name': 'DropoutR', 'type': 'continuous',  'domain': (0.0, 0.5)},
          {'name': 'C1_K',     'type': 'discrete',  'domain': c1_k_range},
          {'name': 'C1_S',     'type': 'discrete',  'domain': c1_s_range},
          {'name': 'C2_K',     'type': 'discrete',  'domain': c2_k_range},
          {'name': 'C2_S',     'type': 'discrete',  'domain': c2_s_range}]

hyper_parameter_names = ["Dense Layer Neuron Count", "Dropout Rate",
                         "Convolutional Filters in Layer 1", "Convolutional Kernel Size in Layer 1",
                         "Convolutional Filters in Layer 2", "Convolutional Kernel Size in Layer 2"]

iter = 0
file = None

# function to optimize the model
def f(x):
    global iter
    iter = iter + 1
    for i in range(6):
        print(f"{hyper_parameter_names[i]}: {x[:,i]}")
    (bestMwd, scores_train_mean, scores_test_mean) = evaluations_of_models(features, labels,  
                                DenseN = int(x[:,0]), 
                                DropoutR = float(x[:,1]), 
                                C1_K=int(x[:,2]), 
                                C1_S=int(x[:,3]), 
                                C2_K=int(x[:,4]), 
                                C2_S=int(x[:,5]),
                                verbose=False)
    evaluation = scores_test_mean['rmse']
    file.write(f"{iter};{scores_train_mean['rmse']};{scores_train_mean['r2']};{scores_train_mean['huber']};{scores_test_mean['rmse']};{scores_test_mean['r2']};{scores_test_mean['huber']};{int(x[:,0])};{float(x[:,1])};{int(x[:,2])};{int(x[:,3])};{int(x[:,4])};{int(x[:,5])}\n")
    print("###############################################")
    print(f"Score of Iteration {iter}: {evaluation}")
    print("###############################################")
    return evaluation

with open('history.csv', 'w') as history_file:
    file = history_file
    print(f'ROUGH TUNING')
    ## Initialize the model with 10 random hyperparameters
    opt_model_gs = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds, initial_design_numdata=10,verbose=1)

    print(f'FINE TUNING')
    iter = 0
    # optimize the model over 200 iteration
    opt_model_gs.run_optimization(max_iter=200)


print("###########################")
print("Optimized Model Parameters:")
print("###########################")

with open('output.txt', 'w') as output_file:
    for i in range(6):
          message = f"{hyper_parameter_names[i]}: {opt_model_gs.x_opt[i]}"
          print(message)
          output_file.write(f"{message}\n")


opt_model_gs.plot_convergence()
plt.savefig('convergence_plot.png')
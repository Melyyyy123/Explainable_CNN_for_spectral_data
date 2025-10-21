07-9_model_final_sub.ipynb

# LeNet 4M Params, 3 min training (Norbert) (NO scheduler etc)

R2    (Train/Test) = 0.689 / 0.621
RMSE  (Train/Test) = 32.994 / 34.311
Huber (Train/Test) = 26.278 / 27.024
Train: rmse (mean of 1 models)= 32.99416082826737 
Test: rmse (mean of 1 models)= 34.31135712473697
Train: r2 (mean of 1 models)= 0.6886838348519114 
Test: r2 (mean of 1 models)= 0.6214365836485638
Train: huber (mean of 1 models)= 26.277772903442383 
Test: huber (mean of 1 models)= 27.023645401000977

R2    = 0.683
RMSE  = 33.128
Huber = 26.352

# LeNet 4M Params, 3 min training (Norbert) (WITH scheduler etc)

R2    (Train/Test) = 0.801 / 0.762
RMSE  (Train/Test) = 26.376 / 27.234
Huber (Train/Test) = 18.699 / 19.927
Train: rmse (mean of 1 models)= 26.37582544405044 
Test: rmse (mean of 1 models)= 27.23400696003154
Train: r2 (mean of 1 models)= 0.8010519814911123 
Test: r2 (mean of 1 models)= 0.7615013427987231
Train: huber (mean of 1 models)= 18.699207305908203 
Test: huber (mean of 1 models)= 19.926794052124023

R2    = 0.798
RMSE  = 26.463
Huber = 18.822


# IncepResNet 500k Params, 4 min training (WITH scheduler etc)

R2    (Train/Test) = 0.953 / 0.826
RMSE  (Train/Test) = 12.813 / 23.240
Huber (Train/Test) = 8.822 / 16.883
Train: rmse (mean of 1 models)= 12.813152402687924 
Test: rmse (mean of 1 models)= 23.24042485249309
Train: r2 (mean of 1 models)= 0.9530495800035634 
Test: r2 (mean of 1 models)= 0.8263195338706892
Train: huber (mean of 1 models)= 8.82162094116211 
Test: huber (mean of 1 models)= 16.883237838745117

R2    = 0.942
RMSE  = 14.205
Huber = 9.628


# IncepResNet 500k Params, 4 min training (WITH scheduler etc, scaling +1800)

R2    (Train/Test) = 0.992 / 0.752
RMSE  (Train/Test) = 5.338 / 27.795
Huber (Train/Test) = 3.414 / 18.483
Train: rmse (mean of 1 models)= 5.338090826130506 
Test: rmse (mean of 1 models)= 27.79517744353303
Train: r2 (mean of 1 models)= 0.9918510919779604 
Test: r2 (mean of 1 models)= 0.7515713085590963
Train: huber (mean of 1 models)= 3.413994789123535 
Test: huber (mean of 1 models)= 18.48259735107422

R2    = 0.970
RMSE  = 10.144
Huber = 4.921

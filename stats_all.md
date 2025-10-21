# LeNet 8M Params, 4 min training, DropoutR=0.0, 

R2    (Train/Test) = -77.334 / -72.897
RMSE  (Train/Test) = 1749.571 / 1748.881
Huber (Train/Test) = 1737.867 / 1736.508
Train: rmse (mean of 5 models)= 404.2161865234375 
Test: rmse (mean of 5 models)= 415.42938232421875
Train: r2 (mean of 5 models)= -14.760440854571272 
Test: r2 (mean of 5 models)= -13.917930907913208
Train: huber (mean of 5 models)= 381.09521484375 
Test: huber (mean of 5 models)= 392.1058654785156

R2    = 0.884
RMSE  = 67.641
Huber = 44.149

# IncepResNet larger kernel 31, 
R2    = 0.934
RMSE  = 50.898
Huber = 35.013

# IncepResNet larger kernels 31, DropoutR=0.2, module 7,15,31
R2    = 0.963
RMSE  = 38.280
Huber = 25.905

# IncepResNet 4M Params, 12 min training, larger kernels 63, DropoutR=0.2, module 15,63,255
R2    = 0.981
RMSE  = 27.542
Huber = 14.935

# IncepResNet 4M Params, 12 min training, larger kernels 63, DropoutR=0.1, module 15,63,255
all_data/tb_logs/incepresnet_all/20251017-175613/it_5/

R2    (Train/Test) = 0.995 / 0.880
RMSE  (Train/Test) = 13.403 / 70.570
Huber (Train/Test) = 9.831 / 45.034
Train: rmse (mean of 5 models)= 12.700809478759766 
Test: rmse (mean of 5 models)= 62.19557571411133
Train: r2 (mean of 5 models)= 0.99577215425767 
Test: r2 (mean of 5 models)= 0.9014227458315377
Train: huber (mean of 5 models)= 9.340642929077148 
Test: huber (mean of 5 models)= 40.218017578125
Best model is from iteration 4

R2    = 0.982
RMSE  = 26.766
Huber = 14.892


# IncepResNet 4M Params, 12 min training, larger kernels 63, DropoutR=0.0, module 15,63,255
all_data/tb_logs/incepresnet_all/20251017-190030/it_4/

R2    (Train/Test) = 1.000 / 0.887
RMSE  (Train/Test) = 3.499 / 68.382
Huber (Train/Test) = 2.031 / 43.629
Train: rmse (mean of 5 models)= 7.013492584228516 
Test: rmse (mean of 5 models)= 61.01628494262695
Train: r2 (mean of 5 models)= 0.9985395270893829 
Test: r2 (mean of 5 models)= 0.9053118730726375
Train: huber (mean of 5 models)= 5.297028541564941 
Test: huber (mean of 5 models)= 39.334556579589844
Best model is from iteration 4

R2    = 0.984
RMSE  = 24.990
Huber = 11.019


# DilutedIncepResNet 0.9M Params, 6 min training, larger kernels 63, DropoutR=0.0, module 15,15,15, Diluted 1,4,16
all_data/tb_logs/incepresnet_all_diluted/20251018-140302/it_1/train 

R2    (Train/Test) = 0.999 / 0.864
RMSE  (Train/Test) = 5.860 / 75.010
Huber (Train/Test) = 2.816 / 50.606
Train: rmse (mean of 5 models)= 15.60316276550293 
Test: rmse (mean of 5 models)= 67.28787994384766
Train: r2 (mean of 5 models)= 0.9897852843063017 
Test: r2 (mean of 5 models)= 0.8849882757681954
Train: huber (mean of 5 models)= 9.957168579101562 
Test: huber (mean of 5 models)= 44.82682418823242

R2    = 0.979
RMSE  = 28.862
Huber = 15.582

# DilutedIncepResNet 1.7M Params, 7 min training, larger kernels 63, DropoutR=0.0, module 15,31,63, Diluted 1,2,4
all_data/tb_logs/incepresnet_all_diluted/20251018-144026/it_1/train

R2    (Train/Test) = 0.999 / 0.867
RMSE  (Train/Test) = 6.898 / 74.289
Huber (Train/Test) = 5.542 / 47.993
Train: rmse (mean of 5 models)= 9.5674409866333 
Test: rmse (mean of 5 models)= 65.58460998535156
Train: r2 (mean of 5 models)= 0.9974949831503773 
Test: r2 (mean of 5 models)= 0.890071067637243
Train: huber (mean of 5 models)= 6.698665618896484 
Test: huber (mean of 5 models)= 41.7591438293457
Best model is from iteration 4

R2    = 0.980
RMSE  = 27.782
Huber = 15.403

# DilutedIncepResNet 1.7M Params, 7 min training, larger kernels 63, DropoutR=0.0, module 15,31,63, Diluted 1,2,4, mixed precision
all_data/tb_logs/incepresnet_all_diluted_mp16/20251018-1532220/it_x/train

R2    (Train/Test) = 0.999 / 0.880
RMSE  (Train/Test) = 5.493 / 70.519
Huber (Train/Test) = 3.395 / 46.149
Train: rmse (mean of 5 models)= 8.70803451538086 
Test: rmse (mean of 5 models)= 65.06111907958984
Train: r2 (mean of 5 models)= 0.9978126624552797 
Test: r2 (mean of 5 models)= 0.8919788123928166
Train: huber (mean of 5 models)= 5.797118186950684 
Test: huber (mean of 5 models)= 42.111080169677734
Best model is from iteration 1

R2    = 0.983
RMSE  = 25.969
Huber = 10.596

# DilutedIncepResNet 1.7M Params, 7 min training, larger kernels 63, DropoutR=0.0, module 15,31,63, Diluted 1,2,4, mixed precision, batch size 128
all_data/tb_logs/incepresnet_all_diluted_mp16_bs128/20251018-163519/it_5/validation 

R2    (Train/Test) = 0.997 / 0.865
RMSE  (Train/Test) = 10.555 / 74.627
Huber (Train/Test) = 7.878 / 49.131
Train: rmse (mean of 5 models)= 10.235844612121582 
Test: rmse (mean of 5 models)= 66.65534973144531
Train: r2 (mean of 5 models)= 0.99733420152212 
Test: r2 (mean of 5 models)= 0.8869407125945561
Train: huber (mean of 5 models)= 7.033358573913574 
Test: huber (mean of 5 models)= 43.71685028076172
Best model is from iteration 1

R2    = 0.980
RMSE  = 27.887
Huber = 13.520


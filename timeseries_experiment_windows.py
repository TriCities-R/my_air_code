import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
import random
import numpy as np
import csv

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='TimesNet')

import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(preds, ground_truth, num_samples=10, filename="prediction_results.png"):
    preds_flat = np.reshape(preds, (-1,))
    ground_truth_flat = np.reshape(ground_truth, (-1,))

    # Plot the first num_samples samples
    plt.figure(figsize=(15, 5))
    plt.plot(ground_truth_flat[:num_samples], label='Ground Truth')
    plt.plot(preds_flat[:num_samples], label='Predictions')
    plt.legend()
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title('Prediction vs Ground Truth')

    # Save the plot as an image file
    plt.savefig(filename)
    plt.close()

def write_results_to_csv(preds, ground_truth, filename="prediction_results.csv"):
    with open(filename, mode="w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Ground Truth", "Prediction"])

        # preds_flat = np.reshape(preds, (-1,))
        # ground_truth_flat = np.reshape(ground_truth, (-1,))

        # for i in range(len(preds_flat)):
        #     writer.writerow([ground_truth_flat[i], preds_flat[i]])
        for i in range(len(preds)):
            writer.writerow([ground_truth[i], preds[i]])



class Args:
    pass

args = Args()






# ETTh1 能用 开始
# Define the desired values for the arguments here
args.task_name = 'long_term_forecast' #options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]
args.is_training = 1
args.model_id = 'test'
args.model = 'Autoformer' # options: [Autoformer, Transformer, TimesNet]'
args.data = 'ETTh1'
args.root_path = './dataset/ETT-small/'
args.data_path = 'ETTh1.csv'
args.features = 'M'  # options:[M, S, MS]
args.target = 'OT'
args.freq = 'h'
args.checkpoints = './checkpoints/'
args.seq_len = 96
args.label_len = 48
args.pred_len = 96
args.seasonal_patterns = 'Monthly'
args.mask_rate = 0.25
args.anomaly_ratio = 0.25
args.top_k = 5
args.num_kernels = 6
args.enc_in = 7
args.dec_in = 7
args.c_out = 7
args.d_model = 512
args.n_heads = 8
args.e_layers = 2
args.d_layers = 1
args.d_ff = 2048
args.moving_avg = 25
args.factor = 1
args.distil = True
args.dropout = 0.1
args.embed = 'timeF'
args.activation = 'gelu'
args.output_attention = False
args.num_workers = 0
args.itr = 1
args.train_epochs = 1
args.batch_size = 32
args.patience = 3
args.learning_rate = 0.0001
args.des = 'test'
args.loss = 'MSE'
args.lradj = 'type1'
args.use_amp = False
args.use_gpu = True
args.gpu = 0
args.use_multi_gpu = False
args.devices = '0,1,2,3'
args.p_hidden_dims = [128, 128]
args.p_hidden_layers = 2
# ETTh1结束


# m4开始
# Set the model_id and other parameters based on the seasonal pattern.
seasonal_pattern = 'Daily'  # Change this to the desired pattern (e.g., 'Yearly', 'Quarterly', 'Daily', 'Weekly', 'Hourly').

if seasonal_pattern == 'Monthly':
    args.model_id = 'm4_Monthly'
    args.d_model = 32
    args.d_ff = 32
elif seasonal_pattern == 'Yearly':
    args.model_id = 'm4_Yearly'
    args.d_model = 16
    args.d_ff = 32
elif seasonal_pattern == 'Quarterly':
    args.model_id = 'm4_Quarterly'
    args.d_model = 64
    args.d_ff = 64
elif seasonal_pattern == 'Daily':
    args.model_id = 'm4_Daily'
    args.d_model = 16
    args.d_ff = 16
elif seasonal_pattern == 'Weekly':
    args.model_id = 'm4_Weekly'
    args.d_model = 32
    args.d_ff = 32
elif seasonal_pattern == 'Hourly':
    args.model_id = 'm4_Hourly'
    args.d_model = 32
    args.d_ff = 32
else:
    raise ValueError(f"Unsupported seasonal_pattern: {seasonal_pattern}")

# Set the remaining parameters.
args.task_name = 'short_term_forecast'
args.is_training = 1
args.root_path = './dataset/m4'
args.seasonal_patterns = seasonal_pattern
args.model = 'TimesNet'
args.data = 'm4'
args.features = 'M'
args.e_layers = 2
args.d_layers = 1
args.factor = 3
args.enc_in = 1
args.dec_in = 1
args.c_out = 1
args.batch_size = 16
args.top_k = 5
args.des = 'Exp'
args.itr = 1
args.learning_rate = 0.001
args.loss = 'SMAPE'
# m4结束







if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

if args.task_name == 'long_term_forecast':
    Exp = Exp_Long_Term_Forecast
elif args.task_name == 'short_term_forecast':
    Exp = Exp_Short_Term_Forecast
elif args.task_name == 'imputation':
    Exp = Exp_Imputation
elif args.task_name == 'anomaly_detection':
    Exp = Exp_Anomaly_Detection
elif args.task_name == 'classification':
    Exp = Exp_Classification
else:
    Exp = Exp_Long_Term_Forecast

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        # After this line: exp.test(setting)
        preds, ground_truth = exp.test(setting)
        plot_predictions(preds, ground_truth, filename="prediction_results.png")
        write_results_to_csv(preds, ground_truth, filename="prediction_results.csv")

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()



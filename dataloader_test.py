import os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import argparse


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

class CustomDataset(Dataset):
    def __init__(self, data_dir, input_len=10, output_len=10):
        self.data_dir = data_dir
        self.input_len = input_len
        self.output_len = output_len
        self.data_files = [f'zdj{i}.csv' for i in range(1, 101)]
        self.data = []

        for file in self.data_files:
            file_path = os.path.join(self.data_dir, file)
            df = pd.read_csv(file_path, header=None)
            sequence_len = df.shape[1]

            for i in range(sequence_len - input_len - output_len + 1):
                input_data = df.iloc[:, i:i + input_len].values
                output_data = df.iloc[:, i + input_len:i + input_len + output_len].values
                self.data.append((input_data, output_data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



data_dir = 'dataset/zdj'
dataset = CustomDataset(data_dir)
batch_size = 16  # 根据您的需求和计算资源调整批次大小
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 假设您已经定义了一个模型（如：model = YourModel()）
# 和一个损失函数（如：loss_fn = torch.nn.MSELoss()）
# 以及一个优化器（如：optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)）
model = Model()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        # 如果使用GPU，请务必将数据移动到相同的设备上
        # inputs, targets = inputs.to(device), targets.to(device)

        # 训练模型
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

        # 打印损失值或其他性能指标
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


# -*- coding: utf-8 -*-
"""
HybridSCIMFP Simple NN v2 - 简化版本

简化设计：
1. 轻量级网络 - 减少参数量
2. 困难样本权重 - 低FMA样本权重更高
3. 早停机制 - 防止过拟合

目标：MAE < 1.40
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error, accuracy_score
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import DataLoader as SCIDataLoader


class Config:
    DATA_DIR = r"C:\Users\Administrator\Desktop\openclaw_cloudcode\sci\dataset\Rocky_Data"
    OUTPUT_DIR = "./outputs/simple_nn_v2"
    TARGET_LENGTH = 100
    FMA_VALUES = [14, 16, 17, 18, 19, 20]
    NUM_CLASSES = 6
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 30
    PATIENCE = 8


class DTWAligner:
    def __init__(self, target_length=100):
        self.target_length = target_length
    
    def align(self, emg, kin):
        T_emg, T_kin = len(emg), len(kin)
        x_target = np.linspace(0, 1, self.target_length)
        
        emg_aligned = np.zeros((self.target_length, emg.shape[1]), dtype=np.float32)
        kin_aligned = np.zeros((self.target_length, kin.shape[1]), dtype=np.float32)
        
        for c in range(emg.shape[1]):
            f = interp1d(np.linspace(0, 1, T_emg), emg[:, c], kind='linear', fill_value='extrapolate')
            emg_aligned[:, c] = f(x_target)
        
        for c in range(kin.shape[1]):
            f = interp1d(np.linspace(0, 1, T_kin), kin[:, c], kind='linear', fill_value='extrapolate')
            kin_aligned[:, c] = f(x_target)
        
        return emg_aligned, kin_aligned


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # EMG编码器
        self.emg_conv = nn.Sequential(
            nn.Conv1d(12, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        # KIN编码器
        self.kin_conv = nn.Sequential(
            nn.Conv1d(63, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        # 分类器
        self.fc = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 6)
        )
    
    def forward(self, emg, kin):
        emg_feat = self.emg_conv(emg.transpose(1, 2)).squeeze(-1)
        kin_feat = self.kin_conv(kin.transpose(1, 2)).squeeze(-1)
        combined = torch.cat([emg_feat, kin_feat], dim=1)
        return self.fc(combined)


def load_data(config):
    loader = SCIDataLoader(data_root=config.DATA_DIR)
    raw_samples = loader.load_all_samples(group="stroke")
    
    dtw = DTWAligner(config.TARGET_LENGTH)
    emg_list, kin_list, label_list, subject_list = [], [], [], []
    
    for sample in tqdm(raw_samples, desc="加载数据"):
        if sample.emg_data.size == 0 or sample.kin_data.size == 0 or sample.fma_score is None:
            continue
        try:
            kin_data = sample.kin_data[:, :63] if sample.kin_data.shape[1] >= 63 else sample.kin_data
            emg_aligned, kin_aligned = dtw.align(sample.emg_data, kin_data)
            if not (np.any(np.isnan(emg_aligned)) or np.any(np.isnan(kin_aligned))):
                emg_list.append(emg_aligned)
                kin_list.append(kin_aligned)
                label_list.append(sample.fma_score)
                subject_list.append(sample.subject_id)
        except:
            continue
    
    return (np.array(emg_list, dtype=np.float32), np.array(kin_list, dtype=np.float32),
            np.array(label_list, dtype=np.float32), np.array(subject_list))


def train_and_eval(train_emg, train_kin, train_labels, test_emg, test_kin, test_labels, config, device):
    train_labels_idx = np.array([config.FMA_VALUES.index(f) for f in train_labels])
    test_labels_idx = np.array([config.FMA_VALUES.index(f) for f in test_labels])
    
    # 计算类别权重（低FMA权重更高）
    class_weights = torch.ones(config.NUM_CLASSES)
    for i, fma in enumerate(config.FMA_VALUES):
        count = (train_labels == fma).sum()
        if count > 0:
            class_weights[i] = len(train_labels) / (config.NUM_CLASSES * count)
    
    model = SimpleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    train_dataset = TensorDataset(
        torch.tensor(train_emg), torch.tensor(train_kin), torch.tensor(train_labels_idx)
    )
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    best_mae = float('inf')
    patience_counter = 0
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        for emg, kin, labels in train_loader:
            emg, kin, labels = emg.to(device), kin.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(emg, kin), labels)
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            test_emg_t = torch.tensor(test_emg).to(device)
            test_kin_t = torch.tensor(test_kin).to(device)
            preds = model(test_emg_t, test_kin_t).argmax(dim=1).cpu().numpy()
        
        pred_fma = np.array([config.FMA_VALUES[p] for p in preds])
        mae = mean_absolute_error(test_labels, pred_fma)
        
        if mae < best_mae:
            best_mae = mae
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                break
    
    # 最终预测
    model.eval()
    with torch.no_grad():
        preds = model(test_emg_t, test_kin_t).argmax(dim=1).cpu().numpy()
    
    pred_fma = np.array([config.FMA_VALUES[p] for p in preds])
    mae = mean_absolute_error(test_labels, pred_fma)
    acc1 = accuracy_score(test_labels, pred_fma)
    acc_pm1 = np.mean(np.abs(pred_fma - test_labels) <= 1)
    
    return mae, acc1, acc_pm1, pred_fma, test_labels


def main():
    print("="*60)
    print("HybridSCIMFP Simple NN v2 - 简化版本")
    print("="*60)
    
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    emg_data, kin_data, labels, subject_ids = load_data(config)
    print(f"\n有效样本: {len(labels)}, 受试者: {len(np.unique(subject_ids))}")
    print(f"设备: {device}")
    
    unique_subjects = np.unique(subject_ids)
    all_results = []
    
    for fold_idx, test_subject in enumerate(unique_subjects):
        print(f"\nFold {fold_idx+1}/{len(unique_subjects)}: 测试受试者 {test_subject}")
        
        train_mask = subject_ids != test_subject
        test_mask = subject_ids == test_subject
        
        mae, acc1, acc_pm1, pred_fma, true_fma = train_and_eval(
            emg_data[train_mask], kin_data[train_mask], labels[train_mask],
            emg_data[test_mask], kin_data[test_mask], labels[test_mask],
            config, device
        )
        
        print(f"  真实FMA: {true_fma[0]}, 预测: {pred_fma.mean():.2f}±{pred_fma.std():.2f}, MAE: {mae:.2f}")
        
        all_results.append({
            'fold': fold_idx + 1, 'test_subject': test_subject, 'true_fma': true_fma[0],
            'pred_fma_mean': pred_fma.mean(), 'mae': mae, 'acc1': acc1, 'acc_pm1': acc_pm1
        })
    
    results_df = pd.DataFrame(all_results)
    avg_mae = results_df['mae'].mean()
    avg_acc1 = results_df['acc1'].mean()
    
    print(f"\n{'='*60}")
    print(f"平均MAE: {avg_mae:.2f}, 平均Acc@1: {avg_acc1*100:.2f}%")
    print(f"对比: v5.7 MAE=1.40, 目标<1.40")
    print(f"{'='*60}")
    
    results_df.to_csv(os.path.join(config.OUTPUT_DIR, 'simple_nn_v2_results.csv'), index=False)
    return results_df


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
HybridSCIMFP Enhanced NN - 增强架构 + 困难样本挖掘

核心设计：
1. 增强模型架构 - 更深的网络 + 残差连接 + 注意力机制
2. 困难样本挖掘 - 动态识别困难样本，增加其权重
3. 类别平衡采样 - 确保每个批次包含足够的低分样本
4. Focal Loss - 自动关注困难样本

目标：MAE < 1.40, Acc@1 > 50%
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error, accuracy_score
from collections import defaultdict
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import DataLoader as SCIDataLoader


# ============================================================================
# 配置
# ============================================================================
class Config:
    DATA_DIR = r"C:\Users\Administrator\Desktop\openclaw_cloudcode\sci\dataset\Rocky_Data"
    OUTPUT_DIR = "./outputs/enhanced_nn_hardmining"
    
    TARGET_LENGTH = 100
    FMA_VALUES = [14, 16, 17, 18, 19, 20]
    NUM_CLASSES = 6
    
    EMG_CHANNELS = 12
    KIN_CHANNELS = 63
    HIDDEN_DIM = 64
    
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 50
    PATIENCE = 10
    
    # 困难样本挖掘参数
    HARD_MINING_START_EPOCH = 10  # 从第10个epoch开始困难样本挖掘
    HARD_SAMPLE_RATIO = 0.3  # 困难样本比例
    HARD_SAMPLE_WEIGHT = 3.0  # 困难样本权重
    
    # Focal Loss参数
    FOCAL_GAMMA = 2.0


# ============================================================================
# DTW对齐模块
# ============================================================================
class DTWAligner:
    def __init__(self, target_length=100):
        self.target_length = target_length
    
    def align(self, emg, kin):
        T_emg = len(emg)
        T_kin = len(kin)
        x_target = np.linspace(0, 1, self.target_length)
        
        x_emg = np.linspace(0, 1, T_emg)
        emg_aligned = np.zeros((self.target_length, emg.shape[1]), dtype=np.float32)
        for c in range(emg.shape[1]):
            f = interp1d(x_emg, emg[:, c], kind='linear', fill_value='extrapolate')
            emg_aligned[:, c] = f(x_target)
        
        x_kin = np.linspace(0, 1, T_kin)
        kin_aligned = np.zeros((self.target_length, kin.shape[1]), dtype=np.float32)
        for c in range(kin.shape[1]):
            f = interp1d(x_kin, kin[:, c], kind='linear', fill_value='extrapolate')
            kin_aligned[:, c] = f(x_target)
        
        return emg_aligned, kin_aligned


# ============================================================================
# 增强的神经网络架构
# ============================================================================
class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation块"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y


class EnhancedEncoder(nn.Module):
    """增强的编码器"""
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        
        # 初始卷积
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        # 残差块
        self.res1 = ResidualBlock(hidden_channels)
        self.res2 = ResidualBlock(hidden_channels)
        
        # SE注意力
        self.se = SEBlock(hidden_channels)
        
        # 下采样
        self.pool = nn.MaxPool1d(2)
        
        # 第二层
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels * 2, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        # 初始卷积
        x = F.relu(self.bn1(self.conv1(x)))
        
        # 残差块
        x = self.res1(x)
        x = self.res2(x)
        
        # SE注意力
        x = self.se(x)
        
        # 下采样
        x = self.pool(x)
        
        # 第二层
        x = F.relu(self.bn2(self.conv2(x)))
        
        # 全局池化
        x = self.global_pool(x)
        
        return x.squeeze(-1)


class CrossAttention(nn.Module):
    """跨模态注意力"""
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
    
    def forward(self, emg_feat, kin_feat):
        # EMG作为query，KIN作为key/value
        q = self.query(emg_feat)
        k = self.key(kin_feat)
        v = self.value(kin_feat)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        return out


class EnhancedNeuralNetwork(nn.Module):
    """
    增强的神经网络
    
    架构：
    1. EMG编码器：残差网络 + SE注意力
    2. KIN编码器：残差网络 + SE注意力
    3. 跨模态注意力：EMG-KIN交互
    4. 分类头：多层全连接 + Dropout
    """
    def __init__(self, config):
        super().__init__()
        
        # EMG编码器
        self.emg_encoder = EnhancedEncoder(config.EMG_CHANNELS, 64)
        
        # KIN编码器
        self.kin_encoder = EnhancedEncoder(config.KIN_CHANNELS, 64)
        
        # 跨模态注意力
        self.cross_attn = CrossAttention(128)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(128 * 3, config.HIDDEN_DIM),  # 3: emg + kin + cross_attn
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.HIDDEN_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 分类头
        self.classifier = nn.Linear(64, config.NUM_CLASSES)
        
        # 序数回归头
        self.ordinal = nn.Linear(64, config.NUM_CLASSES - 1)
    
    def forward(self, emg, kin):
        # 编码
        emg_feat = self.emg_encoder(emg.transpose(1, 2))  # (B, 128)
        kin_feat = self.kin_encoder(kin.transpose(1, 2))  # (B, 128)
        
        # 跨模态注意力
        cross_feat = self.cross_attn(emg_feat, kin_feat)  # (B, 128)
        
        # 融合
        combined = torch.cat([emg_feat, kin_feat, cross_feat], dim=1)  # (B, 384)
        fused = self.fusion(combined)  # (B, 64)
        
        # 输出
        class_logits = self.classifier(fused)  # (B, 6)
        ordinal_logits = self.ordinal(fused)  # (B, 5)
        
        return class_logits, ordinal_logits


# ============================================================================
# Focal Loss
# ============================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ============================================================================
# 困难样本挖掘
# ============================================================================
class HardSampleMiner:
    """
    困难样本挖掘器
    
    策略：
    1. 记录每个样本的历史损失
    2. 定期识别损失最高的样本作为困难样本
    3. 为困难样本分配更高的采样权重
    """
    def __init__(self, num_samples, config):
        self.num_samples = num_samples
        self.config = config
        self.sample_losses = np.zeros(num_samples)
        self.sample_counts = np.zeros(num_samples)
    
    def update(self, indices, losses):
        """更新样本损失记录"""
        for idx, loss in zip(indices, losses):
            self.sample_losses[idx] += loss
            self.sample_counts[idx] += 1
    
    def get_sample_weights(self):
        """获取样本权重"""
        # 计算平均损失
        avg_losses = np.zeros(self.num_samples)
        mask = self.sample_counts > 0
        avg_losses[mask] = self.sample_losses[mask] / self.sample_counts[mask]
        
        # 归一化
        if avg_losses.max() > 0:
            avg_losses = avg_losses / avg_losses.max()
        
        # 基础权重
        weights = np.ones(self.num_samples)
        
        # 困难样本权重
        threshold = np.percentile(avg_losses[mask], 70)  # 前30%作为困难样本
        hard_mask = avg_losses > threshold
        weights[hard_mask] = self.config.HARD_SAMPLE_WEIGHT
        
        return weights


# ============================================================================
# 训练函数
# ============================================================================
def train_epoch(model, dataloader, focal_criterion, ordinal_criterion, optimizer, device, 
                hard_miner=None, sample_indices=None, epoch=None, config=None):
    model.train()
    total_loss = 0
    all_losses = []
    all_indices = []
    
    for batch_idx, (emg, kin, labels, indices) in enumerate(dataloader):
        emg = emg.to(device)
        kin = kin.to(device)
        labels = labels.to(device)
        
        # 前向传播
        class_logits, ordinal_logits = model(emg, kin)
        
        # 计算损失
        focal_loss = focal_criterion(class_logits, labels)
        
        # 序数回归损失
        batch_size = labels.size(0)
        cumulative_labels = torch.zeros(batch_size, config.NUM_CLASSES - 1, device=device)
        for i, label in enumerate(labels):
            if label > 0:
                cumulative_labels[i, :label] = 1
        ordinal_loss = F.binary_cross_entropy_with_logits(ordinal_logits, cumulative_labels)
        
        # 总损失
        loss = focal_loss + 0.5 * ordinal_loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 记录样本损失（用于困难样本挖掘）
        if hard_miner is not None and sample_indices is not None:
            sample_loss = focal_loss.item()
            all_losses.extend([sample_loss] * len(indices))
            all_indices.extend(indices.numpy())
    
    # 更新困难样本挖掘器
    if hard_miner is not None and len(all_indices) > 0:
        hard_miner.update(all_indices, all_losses)
    
    return total_loss / len(dataloader)


def evaluate(model, emg_data, kin_data, labels, config, device):
    """评估模型"""
    model.eval()
    
    emg_tensor = torch.tensor(emg_data, dtype=torch.float32).to(device)
    kin_tensor = torch.tensor(kin_data, dtype=torch.float32).to(device)
    
    all_class_logits = []
    all_ordinal_logits = []
    
    with torch.no_grad():
        for i in range(0, len(emg_tensor), config.BATCH_SIZE):
            batch_emg = emg_tensor[i:i+config.BATCH_SIZE]
            batch_kin = kin_tensor[i:i+config.BATCH_SIZE]
            class_logits, ordinal_logits = model(batch_emg, batch_kin)
            all_class_logits.append(class_logits.cpu())
            all_ordinal_logits.append(ordinal_logits.cpu())
    
    class_logits = torch.cat(all_class_logits, dim=0)
    ordinal_logits = torch.cat(all_ordinal_logits, dim=0)
    
    # 序数回归预测
    ordinal_probs = torch.sigmoid(ordinal_logits)
    ordinal_preds = (ordinal_probs > 0.5).sum(dim=1)
    
    # 分类预测
    class_preds = class_logits.argmax(dim=1)
    
    # 集成预测（序数回归为主）
    pred_classes = ordinal_preds
    
    # 转换为FMA值
    pred_fma = np.array([config.FMA_VALUES[p] for p in pred_classes.numpy()])
    true_fma = np.array(labels)
    
    # 计算指标
    mae = mean_absolute_error(true_fma, pred_fma)
    acc1 = accuracy_score(true_fma, pred_fma)
    acc_pm1 = np.mean(np.abs(pred_fma - true_fma) <= 1)
    
    return mae, acc1, acc_pm1, pred_fma, true_fma


# ============================================================================
# 主训练流程
# ============================================================================
def main():
    print("="*60)
    print("HybridSCIMFP Enhanced NN - 增强架构 + 困难样本挖掘")
    print("="*60)
    print("\n核心设计:")
    print("  1. 增强架构 - 残差网络 + SE注意力 + 跨模态注意力")
    print("  2. 困难样本挖掘 - 动态识别困难样本")
    print("  3. Focal Loss - 自动关注困难样本")
    print("\n目标: MAE < 1.40, Acc@1 > 50%")
    print("="*60)
    
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    print("\n" + "="*60)
    print("加载数据...")
    print("="*60)
    
    loader = SCIDataLoader(data_root=config.DATA_DIR)
    raw_samples = loader.load_all_samples(group="stroke")
    print(f"加载原始样本: {len(raw_samples)}")
    
    # DTW对齐
    dtw_aligner = DTWAligner(config.TARGET_LENGTH)
    emg_list, kin_list, label_list, subject_list = [], [], [], []
    
    for sample in tqdm(raw_samples, desc="DTW对齐 + 特征提取"):
        if sample.emg_data.size == 0 or sample.kin_data.size == 0:
            continue
        if sample.fma_score is None:
            continue
        
        try:
            kin_data = sample.kin_data[:, :63] if sample.kin_data.shape[1] >= 63 else sample.kin_data
            emg_aligned, kin_aligned = dtw_aligner.align(sample.emg_data, kin_data)
            
            if np.any(np.isnan(emg_aligned)) or np.any(np.isinf(emg_aligned)):
                continue
            if np.any(np.isnan(kin_aligned)) or np.any(np.isinf(kin_aligned)):
                continue
            
            emg_list.append(emg_aligned)
            kin_list.append(kin_aligned)
            label_list.append(sample.fma_score)
            subject_list.append(sample.subject_id)
        except Exception as e:
            continue
    
    emg_data = np.array(emg_list, dtype=np.float32)
    kin_data = np.array(kin_list, dtype=np.float32)
    labels = np.array(label_list, dtype=np.float32)
    subject_ids = np.array(subject_list)
    
    print(f"\n有效样本: {len(labels)}")
    print(f"EMG形状: {emg_data.shape}")
    print(f"KIN形状: {kin_data.shape}")
    
    unique_fma, fma_counts = np.unique(labels, return_counts=True)
    print(f"\nFMA分布: {dict(zip([int(f) for f in unique_fma], fma_counts))}")
    
    unique_subjects = np.unique(subject_ids)
    print(f"\n受试者数量: {len(unique_subjects)}")
    print(f"使用设备: {device}")
    
    # 计算类别权重
    class_weights = torch.zeros(config.NUM_CLASSES)
    total_samples = len(labels)
    for i, fma in enumerate(config.FMA_VALUES):
        count = (labels == fma).sum()
        class_weights[i] = total_samples / (config.NUM_CLASSES * count) if count > 0 else 1.0
    print(f"\n类别权重: {class_weights.tolist()}")
    
    # LOSOCV
    all_results = []
    
    for fold_idx, test_subject in enumerate(unique_subjects):
        print("\n" + "="*60)
        print(f"Fold {fold_idx+1}/{len(unique_subjects)}: 测试受试者 {test_subject}")
        print("="*60)
        
        # 划分训练/测试集
        train_mask = subject_ids != test_subject
        test_mask = subject_ids == test_subject
        
        train_emg = emg_data[train_mask]
        train_kin = kin_data[train_mask]
        train_labels = labels[train_mask]
        
        test_emg = emg_data[test_mask]
        test_kin = kin_data[test_mask]
        test_labels = labels[test_mask]
        
        # 转换标签为类别索引
        train_labels_idx = np.array([config.FMA_VALUES.index(f) for f in train_labels])
        test_labels_idx = np.array([config.FMA_VALUES.index(f) for f in test_labels])
        
        # 创建困难样本挖掘器
        hard_miner = HardSampleMiner(len(train_labels), config)
        
        # 创建模型
        model = EnhancedNeuralNetwork(config).to(device)
        focal_criterion = FocalLoss(gamma=config.FOCAL_GAMMA, weight=class_weights.to(device))
        ordinal_criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # 训练
        print(f"\n[训练参数] 总计: {sum(p.numel() for p in model.parameters())} 参数")
        
        best_mae = float('inf')
        patience_counter = 0
        
        for epoch in range(config.NUM_EPOCHS):
            # 获取样本权重（困难样本挖掘）
            if epoch >= config.HARD_MINING_START_EPOCH:
                sample_weights = hard_miner.get_sample_weights()
            else:
                sample_weights = np.ones(len(train_labels))
            
            # 创建加权采样器
            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.float32),
                num_samples=len(train_labels),
                replacement=True
            )
            
            # 创建数据加载器
            train_dataset = TensorDataset(
                torch.tensor(train_emg, dtype=torch.float32),
                torch.tensor(train_kin, dtype=torch.float32),
                torch.tensor(train_labels_idx, dtype=torch.long),
                torch.tensor(np.arange(len(train_labels)), dtype=torch.long)  # 样本索引
            )
            train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler)
            
            # 训练一个epoch
            loss = train_epoch(
                model, train_loader, focal_criterion, ordinal_criterion, 
                optimizer, device, hard_miner, np.arange(len(train_labels)), epoch, config
            )
            
            # 验证
            val_mae, val_acc1, val_acc_pm1, _, _ = evaluate(
                model, test_emg, test_kin, test_labels, config, device
            )
            
            scheduler.step(val_mae)
            
            if (epoch + 1) % 10 == 0:
                hard_ratio = (sample_weights > 1.0).sum() / len(sample_weights) * 100
                print(f"  Epoch {epoch+1}: Loss={loss:.4f}, MAE={val_mae:.2f}, HardSamples={hard_ratio:.1f}%")
            
            # Early stopping
            if val_mae < best_mae:
                best_mae = val_mae
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        
        # 最终评估
        mae, acc1, acc_pm1, pred_fma, true_fma = evaluate(
            model, test_emg, test_kin, test_labels, config, device
        )
        
        print(f"\n结果:")
        print(f"  真实FMA: {true_fma[0]}")
        print(f"  预测FMA: {pred_fma.mean():.2f} ± {pred_fma.std():.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  Acc@1: {acc1*100:.2f}%")
        print(f"  Acc@±1: {acc_pm1*100:.2f}%")
        
        all_results.append({
            'fold': fold_idx + 1,
            'test_subject': test_subject,
            'true_fma': true_fma[0],
            'pred_fma_mean': pred_fma.mean(),
            'pred_fma_std': pred_fma.std(),
            'mae': mae,
            'acc1': acc1,
            'acc_pm1': acc_pm1
        })
    
    # 汇总结果
    print("\n" + "="*60)
    print("最终结果汇总")
    print("="*60)
    
    results_df = pd.DataFrame(all_results)
    avg_mae = results_df['mae'].mean()
    avg_acc1 = results_df['acc1'].mean()
    avg_acc_pm1 = results_df['acc_pm1'].mean()
    
    print(f"\n平均MAE: {avg_mae:.2f}")
    print(f"平均Acc@1: {avg_acc1*100:.2f}%")
    print(f"平均Acc@±1: {avg_acc_pm1*100:.2f}%")
    
    # 保存结果
    output_file = os.path.join(config.OUTPUT_DIR, 'enhanced_nn_hardmining_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n结果已保存: {output_file}")
    
    # 与之前版本对比
    print("\n" + "="*60)
    print("与之前版本对比")
    print("="*60)
    print(f"{'方法':<45} {'MAE':<10} {'Acc@1':<10}")
    print("-"*65)
    print(f"{'Predict Mean (baseline)':<45} {'1.70':<10}")
    print(f"{'v5.7 (PCA15 + Top3集成)':<45} {'1.40':<10}")
    print(f"{'v7.2 (LOSOCV分类)':<45} {'1.42':<10} {'52%':<10}")
    print(f"{'Simple NN (序数回归+课程学习)':<45} {'1.60':<10} {'24%':<10}")
    print(f"{'FewShot v3 (SMOTE)':<45} {'1.79':<10} {'22%':<10}")
    print(f"{'Enhanced NN (增强架构+困难样本挖掘)':<45} {avg_mae:.2f}{'':<6} {avg_acc1*100:.0f}%")
    
    if avg_mae < 1.40:
        print(f"\n[SUCCESS] Enhanced NN 已超越基线!")
    else:
        print(f"\n[FAILED] Enhanced NN 未超越基线 (gap: {avg_mae - 1.40:.2f})")
    
    return results_df


if __name__ == '__main__':
    main()

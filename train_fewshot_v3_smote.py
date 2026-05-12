# -*- coding: utf-8 -*-
"""
HybridSCIMFP FewShot v3 - SMOTE过采样版本

优化内容：
1. SMOTE过采样 - 平衡低FMA类别样本
2. 类别加权损失 - 低分样本权重×2-3
3. Focal Loss - 自动关注困难样本

目标：MAE < 1.40, Acc@1 > 50%
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
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
    OUTPUT_DIR = "./outputs/fewshot_v3_smote"
    
    TARGET_LENGTH = 100
    FMA_VALUES = [14, 16, 17, 18, 19, 20]
    NUM_CLASSES = 6
    
    EMG_CHANNELS = 12
    KIN_CHANNELS = 63
    HIDDEN_DIM = 16
    CNN_CHANNELS = [12, 32, 16]
    KIN_CNN_CHANNELS = [63, 32, 16]
    ATTENTION_HEADS = 2
    
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 10
    PATIENCE = 5

    N_WAY = 6
    K_SHOT = 5
    N_QUERY = 5
    NUM_EPISODES = 10
    
    # SMOTE参数
    SMOTE_TARGET_RATIO = 1.0  # 目标平衡比例（所有类别样本数相同）
    
    # 损失函数权重
    ALPHA_PROTO = 1.0
    ALPHA_TRIPLET = 0.5
    ALPHA_INTRA = 0.1
    MARGIN = 1.0
    
    # 类别权重 - 低分样本权重更高
    CLASS_WEIGHTS = [2.29, 3.21, 2.21, 2.85, 1.0, 1.64]
    
    # Focal Loss参数
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = 0.25


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
# 特征提取器
# ============================================================================
class EMGFeatureExtractor(nn.Module):
    def __init__(self, in_channels=12, hidden_channels=[12, 32, 16]):
        super().__init__()
        layers = []
        for i in range(len(hidden_channels) - 1):
            layers.append(nn.Conv1d(hidden_channels[i], hidden_channels[i+1], 3, padding=1))
            layers.append(nn.BatchNorm1d(hidden_channels[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x)
        return x.squeeze(-1)


class KinematicFeatureExtractor(nn.Module):
    def __init__(self, in_channels=63, hidden_channels=[63, 32, 16]):
        super().__init__()
        layers = []
        for i in range(len(hidden_channels) - 1):
            layers.append(nn.Conv1d(hidden_channels[i], hidden_channels[i+1], 3, padding=1))
            layers.append(nn.BatchNorm1d(hidden_channels[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x)
        return x.squeeze(-1)


class AttentionFusion(nn.Module):
    def __init__(self, emg_dim=16, kin_dim=16, hidden_dim=16, num_heads=2):
        super().__init__()
        self.emg_proj = nn.Linear(emg_dim, hidden_dim)
        self.kin_proj = nn.Linear(kin_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, emg_feat, kin_feat):
        emg_proj = self.emg_proj(emg_feat).unsqueeze(1)
        kin_proj = self.kin_proj(kin_feat).unsqueeze(1)
        combined = torch.cat([emg_proj, kin_proj], dim=1)
        attn_out, _ = self.attention(combined, combined, combined)
        out = self.norm(combined + attn_out)
        return out.mean(dim=1)


class FewShotNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emg_encoder = EMGFeatureExtractor(config.EMG_CHANNELS, config.CNN_CHANNELS)
        self.kin_encoder = KinematicFeatureExtractor(config.KIN_CHANNELS, config.KIN_CNN_CHANNELS)
        self.fusion = AttentionFusion(16, 16, config.HIDDEN_DIM, config.ATTENTION_HEADS)
        self.classifier = nn.Linear(config.HIDDEN_DIM, config.NUM_CLASSES)
    
    def forward(self, emg, kin, return_embedding=False):
        emg_feat = self.emg_encoder(emg)
        kin_feat = self.kin_encoder(kin)
        embedding = self.fusion(emg_feat, kin_feat)
        logits = self.classifier(embedding)
        if return_embedding:
            return embedding, logits
        return logits


# ============================================================================
# 改进的损失函数
# ============================================================================
class ImprovedFewShotLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.alpha_proto = config.ALPHA_PROTO
        self.alpha_triplet = config.ALPHA_TRIPLET
        self.margin = config.MARGIN
        self.num_classes = config.NUM_CLASSES
        
        self.register_buffer('class_weights', torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32))
        self.focal_gamma = config.FOCAL_GAMMA
        self.focal_alpha = config.FOCAL_ALPHA
    
    def weighted_cross_entropy(self, logits, labels):
        weights = self.class_weights[labels]
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        weighted_loss = ce_loss * weights
        return weighted_loss.mean()
    
    def focal_loss(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def ordinal_regression_loss(self, logits, labels):
        batch_size = labels.size(0)
        cumulative_labels = torch.zeros(batch_size, self.num_classes - 1, device=labels.device)
        for i, label in enumerate(labels):
            if label > 0:
                cumulative_labels[i, :label] = 1
        
        loss = 0
        for j in range(self.num_classes - 1):
            loss += F.binary_cross_entropy_with_logits(
                logits[:, j], cumulative_labels[:, j]
            )
        return loss / (self.num_classes - 1)
    
    def prototypical_loss(self, embeddings, labels, prototypes):
        dists = torch.cdist(embeddings, prototypes)
        log_p = F.log_softmax(-dists, dim=1)
        loss = F.nll_loss(log_p, labels)
        return loss
    
    def triplet_loss(self, embeddings, labels):
        batch_size = embeddings.size(0)
        if batch_size < 3:
            return torch.tensor(0.0, device=embeddings.device)
        
        dists = torch.cdist(embeddings, embeddings)
        loss = torch.tensor(0.0, device=embeddings.device)
        count = 0
        
        for i in range(batch_size):
            pos_mask = (labels == labels[i]) & (torch.arange(batch_size, device=labels.device) != i)
            neg_mask = labels != labels[i]
            
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                pos_idx = torch.where(pos_mask)[0]
                neg_idx = torch.where(neg_mask)[0]
                
                pos_d = dists[i, pos_idx].min()
                neg_d = dists[i, neg_idx].min()
                
                loss += F.relu(pos_d - neg_d + self.margin)
                count += 1
        
        return loss / max(count, 1)
    
    def forward(self, embeddings, logits, labels, prototypes):
        loss_ce = self.weighted_cross_entropy(logits, labels)
        loss_focal = self.focal_loss(logits, labels)
        loss_ordinal = self.ordinal_regression_loss(logits, labels)
        loss_proto = self.prototypical_loss(embeddings, labels, prototypes)
        loss_triplet = self.triplet_loss(embeddings, labels)
        
        total_loss = (
            loss_ce +
            0.5 * loss_focal +
            0.3 * loss_ordinal +
            self.alpha_proto * loss_proto +
            self.alpha_triplet * loss_triplet
        )
        
        return total_loss, {
            'ce': loss_ce.item(),
            'focal': loss_focal.item(),
            'ordinal': loss_ordinal.item(),
            'proto': loss_proto.item(),
            'triplet': loss_triplet.item()
        }


# ============================================================================
# SMOTE过采样
# ============================================================================
def apply_smote_oversampling(emg_data, kin_data, labels, config):
    """
    对低FMA类别进行SMOTE过采样
    
    Args:
        emg_data: (N, T, C_emg)
        kin_data: (N, T, C_kin)
        labels: (N,)
        config: 配置对象
    
    Returns:
        过采样后的数据
    """
    print("\n" + "="*60)
    print("SMOTE过采样")
    print("="*60)
    
    # 将时序数据展平为特征向量
    N, T, C_emg = emg_data.shape
    _, _, C_kin = kin_data.shape
    
    # 展平: (N, T*C_emg + T*C_kin)
    emg_flat = emg_data.reshape(N, -1)
    kin_flat = kin_data.reshape(N, -1)
    X = np.concatenate([emg_flat, kin_flat], axis=1)
    
    # 统计原始分布
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\n原始分布:")
    for label, count in zip(unique_labels, counts):
        print(f"  FMA={int(label)}: {count}样本")
    
    # 计算目标样本数（最大类的样本数）
    max_count = counts.max()
    
    # SMOTE过采样
    smote = SMOTE(
        sampling_strategy='auto',  # 自动平衡所有类别
        random_state=42,
        k_neighbors=min(5, counts.min() - 1)  # 确保k_neighbors不超过最小类别样本数
    )
    
    X_resampled, labels_resampled = smote.fit_resample(X, labels)
    
    # 统计过采样后分布
    unique_labels_res, counts_res = np.unique(labels_resampled, return_counts=True)
    print("\n过采样后分布:")
    for label, count in zip(unique_labels_res, counts_res):
        print(f"  FMA={int(label)}: {count}样本")
    
    # 重塑回时序格式
    N_new = X_resampled.shape[0]
    emg_resampled = X_resampled[:, :T*C_emg].reshape(N_new, T, C_emg)
    kin_resampled = X_resampled[:, T*C_emg:].reshape(N_new, T, C_kin)
    
    print(f"\n总样本数: {N} → {N_new}")
    
    return emg_resampled, kin_resampled, labels_resampled


# ============================================================================
# Episode生成器
# ============================================================================
class EpisodeGenerator:
    def __init__(self, emg_data, kin_data, labels, config):
        self.emg_data = emg_data
        self.kin_data = kin_data
        self.labels = labels
        self.config = config
        
        # 按类别分组
        self.class_indices = defaultdict(list)
        for i, label in enumerate(labels):
            self.class_indices[int(label)].append(i)
    
    def generate_episode(self):
        """生成一个episode"""
        support_emg, support_kin, support_labels = [], [], []
        query_emg, query_kin, query_labels = [], [], []
        
        for class_idx, fma_value in enumerate(self.config.FMA_VALUES):
            indices = self.class_indices[fma_value]
            
            if len(indices) < self.config.K_SHOT + self.config.N_QUERY:
                # 如果样本不足，重复采样
                indices = indices * ((self.config.K_SHOT + self.config.N_QUERY) // len(indices) + 1)
            
            perm = np.random.permutation(indices)
            
            # Support set
            for i in range(self.config.K_SHOT):
                idx = perm[i]
                support_emg.append(self.emg_data[idx])
                support_kin.append(self.kin_data[idx])
                support_labels.append(class_idx)
            
            # Query set
            for i in range(self.config.K_SHOT, self.config.K_SHOT + self.config.N_QUERY):
                idx = perm[i]
                query_emg.append(self.emg_data[idx])
                query_kin.append(self.kin_data[idx])
                query_labels.append(class_idx)
        
        return (
            np.array(support_emg), np.array(support_kin), np.array(support_labels),
            np.array(query_emg), np.array(query_kin), np.array(query_labels)
        )


# ============================================================================
# 训练函数
# ============================================================================
def train_epoch(model, episode_gen, criterion, optimizer, config, device):
    model.train()
    total_loss = 0
    loss_details = defaultdict(float)
    
    for episode_idx in range(config.NUM_EPISODES):
        # 生成episode
        s_emg, s_kin, s_labels, q_emg, q_kin, q_labels = episode_gen.generate_episode()
        
        # 转换为tensor
        s_emg = torch.tensor(s_emg, dtype=torch.float32).to(device)
        s_kin = torch.tensor(s_kin, dtype=torch.float32).to(device)
        s_labels = torch.tensor(s_labels, dtype=torch.long).to(device)
        q_emg = torch.tensor(q_emg, dtype=torch.float32).to(device)
        q_kin = torch.tensor(q_kin, dtype=torch.float32).to(device)
        q_labels = torch.tensor(q_labels, dtype=torch.long).to(device)
        
        # 前向传播
        s_embed, s_logits = model(s_emg, s_kin, return_embedding=True)
        q_embed, q_logits = model(q_emg, q_kin, return_embedding=True)
        
        # 计算原型
        prototypes = []
        for c in range(config.NUM_CLASSES):
            mask = s_labels == c
            if mask.sum() > 0:
                proto = s_embed[mask].mean(dim=0)
            else:
                proto = torch.zeros(config.HIDDEN_DIM, device=device)
            prototypes.append(proto)
        prototypes = torch.stack(prototypes)
        
        # 计算损失
        loss, details = criterion(q_embed, q_logits, q_labels, prototypes)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        for k, v in details.items():
            loss_details[k] += v
    
    n = config.NUM_EPISODES
    return total_loss / n, {k: v/n for k, v in loss_details.items()}


def evaluate(model, emg_data, kin_data, labels, config, device):
    """评估模型"""
    model.eval()
    
    # 转换为tensor
    emg_tensor = torch.tensor(emg_data, dtype=torch.float32).to(device)
    kin_tensor = torch.tensor(kin_data, dtype=torch.float32).to(device)
    
    # 批量预测
    predictions = []
    with torch.no_grad():
        for i in range(0, len(emg_tensor), config.BATCH_SIZE):
            batch_emg = emg_tensor[i:i+config.BATCH_SIZE]
            batch_kin = kin_tensor[i:i+config.BATCH_SIZE]
            logits = model(batch_emg, batch_kin)
            preds = logits.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)
    
    predictions = np.array(predictions)
    
    # 转换为FMA值
    pred_fma = np.array([config.FMA_VALUES[p] for p in predictions])
    # labels已经是FMA值，直接使用
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
    print("HybridSCIMFP FewShot v3 - SMOTE过采样版本")
    print("="*60)
    print("\n优化内容:")
    print("  1. SMOTE过采样 - 平衡低FMA类别样本")
    print("  2. 类别加权损失 - 低分样本权重×2-3")
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
            # KIN数据取前63维
            kin_data = sample.kin_data[:, :63] if sample.kin_data.shape[1] >= 63 else sample.kin_data
            
            # DTW对齐
            emg_aligned, kin_aligned = dtw_aligner.align(sample.emg_data, kin_data)
            
            # 检查NaN和Inf
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
    
    # 统计FMA分布
    unique_fma, fma_counts = np.unique(labels, return_counts=True)
    print(f"\nFMA分布: {dict(zip([int(f) for f in unique_fma], fma_counts))}")
    
    unique_subjects = np.unique(subject_ids)
    print(f"\n受试者数量: {len(unique_subjects)}")
    print(f"使用设备: {device}")
    
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
        
        # 【关键】对训练集进行SMOTE过采样
        train_emg_smote, train_kin_smote, train_labels_smote = apply_smote_oversampling(
            train_emg, train_kin, train_labels, config
        )
        
        # 创建episode生成器（使用过采样后的数据）
        episode_gen = EpisodeGenerator(train_emg_smote, train_kin_smote, train_labels_smote, config)
        
        # 创建模型
        model = FewShotNetwork(config).to(device)
        criterion = ImprovedFewShotLoss(config)
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        
        # 训练
        print(f"\n[训练参数] 总计: {sum(p.numel() for p in model.parameters())} 参数")
        
        for epoch in range(config.NUM_EPOCHS):
            loss, details = train_epoch(model, episode_gen, criterion, optimizer, config, device)
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}: Loss={loss:.4f} (CE={details['ce']:.4f}, Focal={details['focal']:.4f}, Ordinal={details['ordinal']:.4f})")
        
        # 评估
        mae, acc1, acc_pm1, pred_fma, true_fma = evaluate(model, test_emg, test_kin, test_labels, config, device)
        
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
    output_file = os.path.join(config.OUTPUT_DIR, 'fewshot_v3_smote_results.csv')
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
    print(f"{'FewShot v1 (原始)':<45} {'1.77':<10} {'31%':<10}")
    print(f"{'FewShot v2 Optimized (阶段1优化)':<45} {'1.86':<10} {'21%':<10}")
    print(f"{'FewShot v3 SMOTE (SMOTE过采样)':<45} {avg_mae:.2f}{'':<6} {avg_acc1*100:.0f}%")
    
    if avg_mae < 1.40:
        print(f"\n[SUCCESS] FewShot v3 SMOTE 已超越基线!")
    else:
        print(f"\n[FAILED] FewShot v3 SMOTE 未超越基线 (gap: {avg_mae - 1.40:.2f})")
    
    return results_df


if __name__ == '__main__':
    main()

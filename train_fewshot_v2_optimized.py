# -*- coding: utf-8 -*-
"""
HybridSCIMFP FewShot v2 Optimized - 阶段1优化版本

优化内容（阶段1：快速见效）：
1. B1. 类别加权损失 - 低分样本权重×2-2.5，强制模型关注低分样本
2. B2. Focal Loss - 自动关注困难样本，gamma=2.0
3. E1. 阈值调整 - 高分预测置信度低时降级

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
    OUTPUT_DIR = "./outputs/fewshot_v2_optimized"
    
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

    QUICK_TEST = False
    
    # 损失函数权重
    ALPHA_PROTO = 1.0
    ALPHA_TRIPLET = 0.5
    ALPHA_INTRA = 0.1
    MARGIN = 1.0
    
    # 【新增】类别权重 - 低分样本权重更高
    # FMA分布: 14(199), 16(142), 17(206), 18(160), 19(455), 20(278)
    # 权重 = 总样本数 / (类别数 * 该类样本数)
    CLASS_WEIGHTS = [2.29, 3.21, 2.21, 2.85, 1.0, 1.64]
    
    # 【新增】Focal Loss参数
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = 0.25
    
    # 【新增】阈值调整参数
    HIGH_CONFIDENCE_THRESHOLD = 0.6  # 高分预测需要超过此置信度


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
        for i in range(emg.shape[1]):
            f = interp1d(x_emg, emg[:, i], kind='linear', fill_value='extrapolate')
            emg_aligned[:, i] = f(x_target)
        
        x_kin = np.linspace(0, 1, T_kin)
        kin_aligned = np.zeros((self.target_length, kin.shape[1]), dtype=np.float32)
        for i in range(kin.shape[1]):
            f = interp1d(x_kin, kin[:, i], kind='linear', fill_value='extrapolate')
            kin_aligned[:, i] = f(x_target)
        
        return emg_aligned, kin_aligned


# ============================================================================
# 轻量级特征编码器
# ============================================================================
class LightweightCNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        layers = []
        for i in range(len(hidden_channels) - 1):
            layers.extend([
                nn.Conv1d(hidden_channels[i], hidden_channels[i+1], kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_channels[i+1]),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ])
        self.encoder = nn.Sequential(*layers)
        self.out_channels = hidden_channels[-1]
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.encoder(x)
        return x


# ============================================================================
# 时序注意力模块
# ============================================================================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x, attn_weights


# ============================================================================
# 度量学习嵌入网络
# ============================================================================
class MetricEmbeddingNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.emg_encoder = LightweightCNNEncoder(config.EMG_CHANNELS, config.CNN_CHANNELS)
        self.kin_encoder = LightweightCNNEncoder(config.KIN_CHANNELS, config.KIN_CNN_CHANNELS)
        
        self.emg_attention = TemporalAttention(config.CNN_CHANNELS[-1], config.ATTENTION_HEADS)
        self.kin_attention = TemporalAttention(config.KIN_CNN_CHANNELS[-1], config.ATTENTION_HEADS)
        
        self.time_dim = config.TARGET_LENGTH // 4
        self.feature_dim = config.CNN_CHANNELS[-1] + config.KIN_CNN_CHANNELS[-1]
        
        self.embedding = nn.Sequential(
            nn.Linear(self.feature_dim * self.time_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, config.HIDDEN_DIM)
        )
        
        # 【新增】分类头
        self.classifier = nn.Linear(config.HIDDEN_DIM, config.NUM_CLASSES)
        
        self._count_parameters()
    
    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"[网络参数量] 总计: {total:,} 个参数")
    
    def forward(self, emg, kin, return_logits=False):
        emg_feat = self.emg_encoder(emg).transpose(1, 2)
        kin_feat = self.kin_encoder(kin).transpose(1, 2)
        
        emg_attn, _ = self.emg_attention(emg_feat)
        kin_attn, _ = self.kin_attention(kin_feat)
        
        emg_flat = emg_attn.reshape(emg_attn.size(0), -1)
        kin_flat = kin_attn.reshape(kin_attn.size(0), -1)
        
        combined = torch.cat([emg_flat, kin_flat], dim=1)
        embedding = self.embedding(combined)
        
        if return_logits:
            logits = self.classifier(embedding)
            return embedding, logits
        return embedding


# ============================================================================
# 【优化】改进的损失函数
# ============================================================================
class ImprovedFewShotLoss(nn.Module):
    """
    改进的小样本学习损失函数
    
    优化内容：
    1. 类别加权交叉熵 - 低分样本权重更高
    2. Focal Loss - 自动关注困难样本
    3. 序数回归损失 - 利用FMA有序性
    """
    def __init__(self, config):
        super().__init__()
        self.alpha_proto = config.ALPHA_PROTO
        self.alpha_triplet = config.ALPHA_TRIPLET
        self.alpha_intra = config.ALPHA_INTRA
        self.margin = config.MARGIN
        self.num_classes = config.NUM_CLASSES
        
        # 【新增】类别权重
        self.register_buffer('class_weights', torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32))
        
        # 【新增】Focal Loss参数
        self.focal_gamma = config.FOCAL_GAMMA
        self.focal_alpha = config.FOCAL_ALPHA
    
    def weighted_cross_entropy(self, logits, labels):
        """
        【优化1】类别加权交叉熵损失
        低分样本权重更高，强制模型关注低分样本
        """
        # 获取每个样本的类别权重
        weights = self.class_weights[labels]
        
        # 计算交叉熵
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        
        # 加权
        weighted_loss = ce_loss * weights
        
        return weighted_loss.mean()
    
    def focal_loss(self, logits, labels):
        """
        【优化2】Focal Loss
        自动关注困难样本（预测概率低的样本）
        
        公式：FL = -α(1-pt)^γ * log(pt)
        """
        # 计算交叉熵
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        
        # 计算预测概率
        pt = torch.exp(-ce_loss)
        
        # Focal Loss
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        
        return focal_loss.mean()
    
    def ordinal_regression_loss(self, logits, labels):
        """
        【优化3】序数回归损失
        利用FMA的有序性（14 < 16 < 17 < 18 < 19 < 20）
        
        将分类问题转换为多个二分类问题
        """
        batch_size = labels.size(0)
        
        # 创建累积标签
        # FMA=14 → [0,0,0,0,0]
        # FMA=16 → [1,0,0,0,0]
        # FMA=17 → [1,1,0,0,0]
        # ...
        cumulative_labels = torch.zeros(batch_size, self.num_classes - 1, device=labels.device)
        for i, label in enumerate(labels):
            if label > 0:
                cumulative_labels[i, :label] = 1
        
        # 对每个阈值进行二分类
        loss = 0
        for j in range(self.num_classes - 1):
            loss += F.binary_cross_entropy_with_logits(
                logits[:, j], cumulative_labels[:, j]
            )
        
        return loss / (self.num_classes - 1)
    
    def prototypical_loss(self, embeddings, labels, prototypes):
        """原型对比损失"""
        dists = torch.cdist(embeddings, prototypes)
        log_p = F.log_softmax(-dists, dim=1)
        loss = F.nll_loss(log_p, labels)
        return loss
    
    def triplet_loss(self, embeddings, labels):
        """三元组损失"""
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
        """
        总损失
        
        组合多种损失：
        1. 类别加权交叉熵（权重1.0）
        2. Focal Loss（权重0.5）
        3. 序数回归损失（权重0.3）
        4. 原型对比损失（权重1.0）
        5. 三元组损失（权重0.5）
        """
        # 1. 类别加权交叉熵
        loss_ce = self.weighted_cross_entropy(logits, labels)
        
        # 2. Focal Loss
        loss_focal = self.focal_loss(logits, labels)
        
        # 3. 序数回归损失
        loss_ordinal = self.ordinal_regression_loss(logits, labels)
        
        # 4. 原型对比损失
        loss_proto = self.prototypical_loss(embeddings, labels, prototypes)
        
        # 5. 三元组损失
        loss_triplet = self.triplet_loss(embeddings, labels)
        
        # 总损失
        total_loss = (1.0 * loss_ce + 
                      0.5 * loss_focal + 
                      0.3 * loss_ordinal +
                      self.alpha_proto * loss_proto + 
                      self.alpha_triplet * loss_triplet)
        
        return total_loss, loss_ce, loss_focal, loss_ordinal


# ============================================================================
# Episode采样器
# ============================================================================
class EpisodeSampler:
    def __init__(self, X, y, subjects, config):
        self.X = X
        self.y = y
        self.subjects = subjects
        self.n_way = config.N_WAY
        self.k_shot = config.K_SHOT
        self.n_query = config.N_QUERY
        self.fma_values = config.FMA_VALUES
        
        self.class_indices = defaultdict(list)
        for i, fma in enumerate(y):
            self.class_indices[fma].append(i)
    
    def sample_episode(self):
        available_classes = [c for c in self.class_indices.keys() 
                           if len(self.class_indices[c]) >= self.k_shot + self.n_query]
        
        if len(available_classes) < self.n_way:
            available_classes = list(self.class_indices.keys())
        
        selected_classes = np.random.choice(available_classes, self.n_way, replace=False)
        
        support_indices = []
        query_indices = []
        
        for cls in selected_classes:
            indices = self.class_indices[cls]
            np.random.shuffle(indices)
            
            support_indices.extend(indices[:self.k_shot])
            query_indices.extend(indices[self.k_shot:self.k_shot + self.n_query])
        
        return support_indices, query_indices, selected_classes


# ============================================================================
# 数据加载与预处理
# ============================================================================
def load_and_preprocess(config):
    print("=" * 60)
    print("加载数据...")
    print("=" * 60)
    
    loader = SCIDataLoader(data_root=config.DATA_DIR)
    raw_samples = loader.load_all_samples(group="stroke")
    print(f"加载原始样本: {len(raw_samples)}")
    
    dtw_aligner = DTWAligner(target_length=config.TARGET_LENGTH)
    
    X_emg = []
    X_kin = []
    y = []
    subjects = []
    
    print("DTW对齐 + 特征提取...")
    for sample in tqdm(raw_samples):
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
            
            X_emg.append(emg_aligned)
            X_kin.append(kin_aligned)
            y.append(sample.fma_score)
            subjects.append(sample.subject_id)
            
        except Exception as e:
            continue
    
    X_emg = np.array(X_emg, dtype=np.float32)
    X_kin = np.array(X_kin, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    subjects = np.array(subjects)
    
    print(f"\n有效样本: {len(y)}")
    print(f"EMG形状: {X_emg.shape}")
    print(f"KIN形状: {X_kin.shape}")
    print(f"FMA分布: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X_emg, X_kin, y, subjects


# ============================================================================
# 【优化】阈值调整后处理
# ============================================================================
def adjust_predictions_with_threshold(pred_probs, pred_indices, config):
    """
    【优化3】阈值调整
    
    如果预测高分（FMA=19或20）但置信度低，则考虑降级到第二候选
    
    这解决了模型倾向于预测高分的问题
    """
    adjusted_preds = []
    
    for i, (probs, pred_idx) in enumerate(zip(pred_probs, pred_indices)):
        # 获取排序后的预测
        sorted_indices = np.argsort(probs)[::-1]
        top1_idx = sorted_indices[0]
        top1_prob = probs[top1_idx]
        
        # 如果预测是高分（FMA=19或20）但置信度低
        if top1_idx >= 4 and top1_prob < config.HIGH_CONFIDENCE_THRESHOLD:
            # 降级到第二候选
            if len(sorted_indices) > 1:
                adjusted_preds.append(sorted_indices[1])
            else:
                adjusted_preds.append(top1_idx)
        else:
            adjusted_preds.append(top1_idx)
    
    return adjusted_preds


# ============================================================================
# 训练函数
# ============================================================================
def train_fewshot_v2_optimized(config):
    print("=" * 60)
    print("HybridSCIMFP FewShot v2 Optimized - 阶段1优化版本")
    print("=" * 60)
    print()
    print("优化内容:")
    print("  1. 类别加权损失 - 低分样本权重×2-3")
    print("  2. Focal Loss - 自动关注困难样本")
    print("  3. 阈值调整 - 高分预测降级")
    print()
    print(f"目标: MAE < 1.40, Acc@1 > 50%")
    print("=" * 60)
    
    X_emg, X_kin, y, subjects = load_and_preprocess(config)
    
    unique_subjects = np.unique(subjects)
    print(f"\n受试者数量: {len(unique_subjects)}")
    
    all_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    if config.QUICK_TEST:
        unique_subjects = unique_subjects[:5]
        print(f"\n[快速测试模式] 只训练前 {len(unique_subjects)} 个fold")
    
    for fold, test_subject in enumerate(unique_subjects):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{len(unique_subjects)}: 测试受试者 {test_subject}")
        print("=" * 60)
        
        train_mask = subjects != test_subject
        test_mask = subjects == test_subject
        
        X_emg_train = X_emg[train_mask]
        X_kin_train = X_kin[train_mask]
        y_train = y[train_mask]
        subjects_train = subjects[train_mask]
        
        X_emg_test = X_emg[test_mask]
        X_kin_test = X_kin[test_mask]
        y_test = y[test_mask]
        
        X_emg_train_t = torch.tensor(X_emg_train, dtype=torch.float32)
        X_kin_train_t = torch.tensor(X_kin_train, dtype=torch.float32)
        y_train_t = torch.tensor([config.FMA_VALUES.index(f) for f in y_train], dtype=torch.long)
        
        X_emg_test_t = torch.tensor(X_emg_test, dtype=torch.float32)
        X_kin_test_t = torch.tensor(X_kin_test, dtype=torch.float32)
        y_test_fma = y_test
        
        # 初始化网络
        model = MetricEmbeddingNet(config).to(device)
        criterion = ImprovedFewShotLoss(config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, 
                               weight_decay=config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=5)
        
        episode_sampler = EpisodeSampler(X_emg_train, y_train, subjects_train, config)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.NUM_EPOCHS):
            model.train()
            epoch_loss = 0
            epoch_ce_loss = 0
            epoch_focal_loss = 0
            epoch_ordinal_loss = 0
            
            for episode in range(config.NUM_EPISODES):
                support_idx, query_idx, selected_classes = episode_sampler.sample_episode()
                
                class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}
                
                emg_support = X_emg_train_t[support_idx].to(device)
                kin_support = X_kin_train_t[support_idx].to(device)
                y_support_raw = y_train_t[support_idx].numpy()
                
                emg_query = X_emg_train_t[query_idx].to(device)
                kin_query = X_kin_train_t[query_idx].to(device)
                y_query_raw = y_train_t[query_idx].numpy()
                
                y_support_mapped = torch.tensor(
                    [class_to_idx[config.FMA_VALUES[idx]] for idx in y_support_raw], 
                    dtype=torch.long, device=device
                )
                y_query_mapped = torch.tensor(
                    [class_to_idx[config.FMA_VALUES[idx]] for idx in y_query_raw], 
                    dtype=torch.long, device=device
                )
                
                # 前向传播（返回嵌入和logits）
                support_embed, support_logits = model(emg_support, kin_support, return_logits=True)
                query_embed, query_logits = model(emg_query, kin_query, return_logits=True)
                
                # 计算原型
                prototypes = []
                for i, cls in enumerate(selected_classes):
                    cls_idx = [j for j, c in enumerate(y_support_raw) if config.FMA_VALUES[c] == cls]
                    if cls_idx:
                        proto = support_embed[cls_idx].mean(dim=0)
                        prototypes.append(proto)
                    else:
                        prototypes.append(torch.zeros(config.HIDDEN_DIM, device=device))
                prototypes = torch.stack(prototypes)
                
                # 计算损失
                loss, loss_ce, loss_focal, loss_ordinal = criterion(
                    query_embed, query_logits, y_query_mapped, prototypes
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_ce_loss += loss_ce.item()
                epoch_focal_loss += loss_focal.item()
                epoch_ordinal_loss += loss_ordinal.item()
            
            avg_loss = epoch_loss / config.NUM_EPISODES
            scheduler.step(avg_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f} "
                      f"(CE={epoch_ce_loss/config.NUM_EPISODES:.4f}, "
                      f"Focal={epoch_focal_loss/config.NUM_EPISODES:.4f}, "
                      f"Ordinal={epoch_ordinal_loss/config.NUM_EPISODES:.4f})")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        
        # 测试
        model.eval()
        with torch.no_grad():
            train_embeds, train_logits = model(X_emg_train_t.to(device), X_kin_train_t.to(device), return_logits=True)
            
            prototypes = []
            for fma in config.FMA_VALUES:
                mask = y_train == fma
                if mask.sum() > 0:
                    proto = train_embeds[mask].mean(dim=0)
                    prototypes.append(proto)
                else:
                    prototypes.append(torch.zeros(config.HIDDEN_DIM, device=device))
            prototypes = torch.stack(prototypes)
            
            test_embeds, test_logits = model(X_emg_test_t.to(device), X_kin_test_t.to(device), return_logits=True)
            
            # 方法1：基于分类头的预测
            pred_probs = F.softmax(test_logits, dim=1).cpu().numpy()
            pred_indices = pred_probs.argmax(axis=1)
            
            # 【优化】阈值调整
            adjusted_indices = adjust_predictions_with_threshold(pred_probs, pred_indices, config)
            
            # 方法2：基于原型的预测
            dists = torch.cdist(test_embeds, prototypes)
            proto_indices = dists.argmin(dim=1).cpu().numpy()
            
            # 综合两种方法：如果分类头和原型预测一致，使用分类头；否则使用原型
            final_preds = []
            for i in range(len(pred_indices)):
                if pred_indices[i] == proto_indices[i]:
                    final_preds.append(adjusted_indices[i])
                else:
                    # 不一致时，选择置信度更高的
                    if pred_probs[i].max() > 0.5:
                        final_preds.append(adjusted_indices[i])
                    else:
                        final_preds.append(proto_indices[i])
            
            pred_fma = [config.FMA_VALUES[i] for i in final_preds]
        
        mae = mean_absolute_error(y_test_fma, pred_fma)
        acc1 = accuracy_score(y_test_fma, pred_fma)
        acc_pm1 = sum(1 for p, t in zip(pred_fma, y_test_fma) if abs(p - t) <= 1) / len(y_test_fma)
        
        print(f"\n  结果:")
        print(f"    真实FMA: {y_test_fma[0]}")
        print(f"    预测FMA: {np.mean(pred_fma):.2f} ± {np.std(pred_fma):.2f}")
        print(f"    MAE: {mae:.2f}")
        print(f"    Acc@1: {acc1:.2%}")
        print(f"    Acc@±1: {acc_pm1:.2%}")
        
        all_results.append({
            'fold': fold + 1,
            'test_subject': test_subject,
            'true_fma': y_test_fma[0],
            'pred_fma_mean': np.mean(pred_fma),
            'pred_fma_std': np.std(pred_fma),
            'mae': mae,
            'acc1': acc1,
            'acc_pm1': acc_pm1
        })
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("最终结果汇总")
    print("=" * 60)
    
    results_df = pd.DataFrame(all_results)
    mean_mae = results_df['mae'].mean()
    mean_acc1 = results_df['acc1'].mean()
    mean_acc_pm1 = results_df['acc_pm1'].mean()
    
    print(f"\n平均MAE: {mean_mae:.2f}")
    print(f"平均Acc@1: {mean_acc1:.2%}")
    print(f"平均Acc@±1: {mean_acc_pm1:.2%}")
    
    # 保存结果
    output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    results_df.to_csv(f"{output_dir}/fewshot_v2_optimized_results.csv", index=False)
    
    config_dict = {
        'version': 'fewshot_v2_optimized',
        'timestamp': datetime.now().isoformat(),
        'mean_mae': mean_mae,
        'mean_acc1': mean_acc1,
        'mean_acc_pm1': mean_acc_pm1,
        'optimizations': [
            'Class-weighted CrossEntropy (weights: [2.29, 3.21, 2.21, 2.85, 1.0, 1.64])',
            'Focal Loss (gamma=2.0, alpha=0.25)',
            'Threshold adjustment (high_confidence_threshold=0.6)'
        ],
        'config': {k: v for k, v in vars(config).items() if not k.startswith('_')}
    }
    with open(f"{output_dir}/fewshot_v2_optimized_config.json", 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    print(f"\n结果已保存: {output_dir}/fewshot_v2_optimized_results.csv")
    
    # 对比
    print("\n" + "=" * 60)
    print("与之前版本对比")
    print("=" * 60)
    print(f"{'方法':<45} {'MAE':<10} {'Acc@1':<10}")
    print("-" * 65)
    print(f"{'Predict Mean (baseline)':<45} {1.70:<10.2f}")
    print(f"{'v5.7 (PCA15 + Top3集成)':<45} {1.40:<10.2f}")
    print(f"{'v7.2 (LOSOCV分类)':<45} {1.42:<10.2f} {'52%':<10}")
    print(f"{'FewShot v1 (原始)':<45} {1.77:<10.2f} {'31%':<10}")
    print(f"{'FewShot v2 Optimized (阶段1优化)':<45} {mean_mae:<10.2f} {mean_acc1:.0%}")
    
    if mean_mae < 1.40:
        print("\n[SUCCESS] FewShot v2 Optimized 超越 v5.7!")
    elif mean_mae < 1.70:
        print(f"\n[PROGRESS] FewShot v2 Optimized 接近 v5.7 (gap: {mean_mae - 1.40:.2f})")
    else:
        print(f"\n[FAILED] FewShot v2 Optimized 未超越基线 (gap: {mean_mae - 1.70:.2f})")
    
    return mean_mae, mean_acc1


if __name__ == '__main__':
    config = Config()
    train_fewshot_v2_optimized(config)

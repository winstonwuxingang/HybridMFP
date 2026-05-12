# -*- coding: utf-8 -*-
"""
HybridSCIMFP FewShot v2 - 简化版度量学习网络

核心改进：
1. 移除Episode训练 - 使用标准批量训练，大幅加速
2. 保留度量学习 - 原型对比损失 + 三元组损失
3. 保留时序注意力 - 关注波形最明显特征
4. LOSOCV验证 - 20折交叉验证

目标：MAE < 1.40 (超越v5.7)
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
from torch.utils.data import DataLoader, TensorDataset
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
    OUTPUT_DIR = "./outputs/fewshot_v2"
    
    TARGET_LENGTH = 100
    FMA_VALUES = [14, 16, 17, 18, 19, 20]
    NUM_CLASSES = 6
    
    EMG_CHANNELS = 12
    KIN_CHANNELS = 63
    HIDDEN_DIM = 16
    CNN_CHANNELS = [12, 32, 16]
    KIN_CNN_CHANNELS = [63, 32, 16]
    ATTENTION_HEADS = 2
    
    BATCH_SIZE = 128  # 增大batch size加速
    LEARNING_RATE = 0.002
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 20  # 减少训练轮次
    PATIENCE = 5
    
    # 损失权重
    ALPHA_PROTO = 1.0
    ALPHA_TRIPLET = 0.5
    MARGIN = 1.0


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
        
        emg_aligned = np.zeros((self.target_length, emg.shape[1]), dtype=np.float32)
        for i in range(emg.shape[1]):
            f = interp1d(np.linspace(0, 1, T_emg), emg[:, i], kind='linear', fill_value='extrapolate')
            emg_aligned[:, i] = f(x_target)
        
        kin_aligned = np.zeros((self.target_length, kin.shape[1]), dtype=np.float32)
        for i in range(kin.shape[1]):
            f = interp1d(np.linspace(0, 1, T_kin), kin[:, i], kind='linear', fill_value='extrapolate')
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
        return self.encoder(x)


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
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + self.dropout(attn_out))


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
        
        # 分类头
        self.classifier = nn.Linear(config.HIDDEN_DIM, config.NUM_CLASSES)
        
        self._count_parameters()
    
    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"[网络参数量] 总计: {total:,} 个参数")
    
    def forward(self, emg, kin):
        emg_feat = self.emg_encoder(emg).transpose(1, 2)
        kin_feat = self.kin_encoder(kin).transpose(1, 2)
        
        emg_attn = self.emg_attention(emg_feat)
        kin_attn = self.kin_attention(kin_feat)
        
        emg_flat = emg_attn.reshape(emg_attn.size(0), -1)
        kin_flat = kin_attn.reshape(kin_attn.size(0), -1)
        
        combined = torch.cat([emg_flat, kin_flat], dim=1)
        embedding = self.embedding(combined)
        
        return embedding, self.classifier(embedding)


# ============================================================================
# 损失函数
# ============================================================================
class MetricLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.alpha_proto = config.ALPHA_PROTO
        self.alpha_triplet = config.ALPHA_TRIPLET
        self.margin = config.MARGIN
        self.ce_loss = nn.CrossEntropyLoss()
    
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
                pos_d = dists[i, pos_mask].min()
                neg_d = dists[i, neg_mask].min()
                loss += F.relu(pos_d - neg_d + self.margin)
                count += 1
        
        return loss / max(count, 1)
    
    def forward(self, embeddings, logits, labels, prototypes):
        # 交叉熵损失
        loss_ce = self.ce_loss(logits, labels)
        
        # 三元组损失
        loss_triplet = self.triplet_loss(embeddings, labels)
        
        # 原型对比损失
        loss_proto = torch.tensor(0.0, device=embeddings.device)
        if prototypes is not None:
            dists = torch.cdist(embeddings, prototypes)
            loss_proto = F.nll_loss(F.log_softmax(-dists, dim=1), labels)
        
        return loss_ce + self.alpha_triplet * loss_triplet + self.alpha_proto * loss_proto


# ============================================================================
# 数据加载
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
    
    print("DTW对齐...")
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
        except:
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
# 训练
# ============================================================================
def train_fewshot_v2(config):
    print("=" * 60)
    print("HybridSCIMFP FewShot v2 - 简化版度量学习网络")
    print("=" * 60)
    print("\n核心改进:")
    print("  1. 移除Episode训练 - 标准批量训练，大幅加速")
    print("  2. 保留度量学习 - 原型对比+三元组损失")
    print("  3. 保留时序注意力 - 关注波形最明显特征")
    print(f"\n目标: MAE < 1.40 (超越v5.7)")
    print("=" * 60)
    
    X_emg, X_kin, y, subjects = load_and_preprocess(config)
    
    unique_subjects = np.unique(subjects)
    print(f"\n受试者数量: {len(unique_subjects)}")
    
    all_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    for fold, test_subject in enumerate(unique_subjects):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{len(unique_subjects)}: 测试受试者 {test_subject}")
        print("=" * 60)
        
        train_mask = subjects != test_subject
        test_mask = subjects == test_subject
        
        X_emg_train = X_emg[train_mask]
        X_kin_train = X_kin[train_mask]
        y_train = y[train_mask]
        
        X_emg_test = X_emg[test_mask]
        X_kin_test = X_kin[test_mask]
        y_test = y[test_mask]
        
        # 转换为Tensor
        X_emg_train_t = torch.tensor(X_emg_train, dtype=torch.float32)
        X_kin_train_t = torch.tensor(X_kin_train, dtype=torch.float32)
        y_train_t = torch.tensor([config.FMA_VALUES.index(f) for f in y_train], dtype=torch.long)
        
        X_emg_test_t = torch.tensor(X_emg_test, dtype=torch.float32)
        X_kin_test_t = torch.tensor(X_kin_test, dtype=torch.float32)
        y_test_fma = y_test
        
        # DataLoader
        train_dataset = TensorDataset(X_emg_train_t, X_kin_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        
        # 初始化网络
        model = MetricEmbeddingNet(config).to(device)
        criterion = MetricLoss(config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # 计算初始原型
        def compute_prototypes(model, emg, kin, labels, device):
            model.eval()
            with torch.no_grad():
                embeddings, _ = model(emg.to(device), kin.to(device))
            prototypes = []
            for fma in range(config.NUM_CLASSES):
                mask = labels == fma
                if mask.sum() > 0:
                    proto = embeddings[mask].mean(dim=0)
                    prototypes.append(proto)
                else:
                    prototypes.append(torch.zeros(config.HIDDEN_DIM, device=device))
            return torch.stack(prototypes)
        
        prototypes = compute_prototypes(model, X_emg_train_t, X_kin_train_t, y_train_t, device)
        
        # 训练
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.NUM_EPOCHS):
            model.train()
            epoch_loss = 0
            
            for batch_emg, batch_kin, batch_y in train_loader:
                batch_emg = batch_emg.to(device)
                batch_kin = batch_kin.to(device)
                batch_y = batch_y.to(device)
                
                embeddings, logits = model(batch_emg, batch_kin)
                loss = criterion(embeddings, logits, batch_y, prototypes)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            scheduler.step(avg_loss)
            
            # 更新原型
            if (epoch + 1) % 5 == 0:
                prototypes = compute_prototypes(model, X_emg_train_t, X_kin_train_t, y_train_t, device)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}")
            
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
            # 更新原型
            prototypes = compute_prototypes(model, X_emg_train_t, X_kin_train_t, y_train_t, device)
            
            # 测试集预测
            test_embeddings, test_logits = model(X_emg_test_t.to(device), X_kin_test_t.to(device))
            
            # 方法1: 分类器预测
            pred_indices = test_logits.argmax(dim=1).cpu().numpy()
            pred_fma_classifier = [config.FMA_VALUES[i] for i in pred_indices]
            
            # 方法2: 原型距离预测
            dists = torch.cdist(test_embeddings, prototypes)
            pred_indices_proto = dists.argmin(dim=1).cpu().numpy()
            pred_fma_proto = [config.FMA_VALUES[i] for i in pred_indices_proto]
            
            # 使用分类器预测
            pred_fma = pred_fma_classifier
        
        # 评估
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
    
    # 汇总
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
    
    # 保存
    output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    results_df.to_csv(f"{output_dir}/fewshot_v2_results.csv", index=False)
    
    config_dict = {
        'version': 'fewshot_v2',
        'timestamp': datetime.now().isoformat(),
        'mean_mae': mean_mae,
        'mean_acc1': mean_acc1,
        'mean_acc_pm1': mean_acc_pm1,
        'config': {k: v for k, v in vars(config).items() if not k.startswith('_')}
    }
    with open(f"{output_dir}/fewshot_v2_config.json", 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    print(f"\n结果已保存: {output_dir}/fewshot_v2_results.csv")
    
    # 对比
    print("\n" + "=" * 60)
    print("与之前版本对比")
    print("=" * 60)
    print(f"{'方法':<40} {'MAE':<10} {'Acc@1':<10}")
    print("-" * 60)
    print(f"{'Predict Mean (baseline)':<40} {1.70:<10.2f}")
    print(f"{'v5.7 (PCA15 + Top3集成)':<40} {1.40:<10.2f}")
    print(f"{'v7.2 (LOSOCV分类)':<40} {1.42:<10.2f} {'52%':<10}")
    print(f"{'FewShot v1 (Episode训练)':<40} {1.77:<10.2f} {'31%':<10}")
    print(f"{'FewShot v2 (简化版)':<40} {mean_mae:<10.2f} {mean_acc1:.0%}")
    
    if mean_mae < 1.40:
        print("\n[SUCCESS] FewShot v2 超越 v5.7!")
    elif mean_mae < 1.70:
        print(f"\n[PROGRESS] FewShot v2 接近基线 (gap: {mean_mae - 1.40:.2f})")
    else:
        print(f"\n[FAILED] FewShot v2 未超越基线 (gap: {mean_mae - 1.70:.2f})")
    
    return mean_mae, mean_acc1


if __name__ == '__main__':
    config = Config()
    train_fewshot_v2(config)

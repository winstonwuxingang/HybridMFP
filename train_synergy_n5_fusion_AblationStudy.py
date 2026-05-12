# -*- coding: utf-8 -*-
"""
HybridSCIMFP - 消融实验 (Ablation Study)

针对 train_synergy_n5_fusion.py 的全面消融实验

消融实验分支：
1. 特征消融：
   - ablation=1: 完整模型（基线）
   - ablation=2: 去除层次化融合特征
   - ablation=3: 去除肌肉协同性特征
   - ablation=4: 去除EMG协方差特征
   - ablation=5: 去除EMG频域特征
   - ablation=6: 去除EMG非线性特征
   - ablation=7: 去除KIN特征
   - ablation=8: 仅使用EMG特征
   - ablation=9: 仅使用KIN特征
   - ablation=10: 仅使用融合特征

2. 参数消融：
   - n_synergies: 3,4,5,6,7 (肌肉协同性数量)

3. 方法消融：
   - feature_selection: elastic_net, f_test, none (特征选择方法)
   - dim_reduction: pca, none (降维方法)
   - model: svr, knn, gb, ensemble (预测模型)

使用方法：
    python train_synergy_n5_fusion_AblationStudy.py --ablation 1
    python train_synergy_n5_fusion_AblationStudy.py --ablation all
    python train_synergy_n5_fusion_AblationStudy.py --n_synergies 3
    python train_synergy_n5_fusion_AblationStudy.py --feature_selection f_test
    python train_synergy_n5_fusion_AblationStudy.py --model svr
"""
import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy.signal import welch
from scipy.stats import entropy, kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, NMF
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNetCV
import torch
import torch.nn as nn
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import DataLoader as SCIDataLoader
from fusion.hierarchical_fusion import HierarchicalFusionNetwork


# ============================================================================
# 配置
# ============================================================================
class Config:
    DATA_DIR = r"C:\Users\Administrator\Desktop\openclaw_cloudcode\sci\dataset\Rocky_Data"
    OUTPUT_DIR = "./outputs/ablation_study"
    TARGET_LENGTH = 100
    FMA_VALUES = [14, 16, 17, 18, 19, 20]
    HIDDEN_DIM = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# 消融实验名称映射
# ============================================================================
ABLATION_NAMES = {
    1: "完整模型(基线)",
    2: "去除层次化融合特征",
    3: "去除肌肉协同性特征",
    4: "去除EMG协方差特征",
    5: "去除EMG频域特征",
    6: "去除EMG非线性特征",
    7: "去除KIN特征",
    8: "仅使用EMG特征",
    9: "仅使用KIN特征",
    10: "仅使用融合特征",
}


# ============================================================================
# 基础组件（与原程序相同）
# ============================================================================
class DTWAligner:
    def __init__(self, target_length=100):
        self.target_length = target_length
    
    def align(self, emg, kin):
        T_emg = len(emg)
        T_kin = len(kin)
        x_target = np.linspace(0, 1, self.target_length)
        
        x_emg = np.linspace(0, 1, T_emg)
        emg_aligned = np.zeros((self.target_length, emg.shape[1]))
        for i in range(emg.shape[1]):
            f = interp1d(x_emg, emg[:, i], kind='linear', fill_value='extrapolate')
            emg_aligned[:, i] = f(x_target)
        
        x_kin = np.linspace(0, 1, T_kin)
        kin_aligned = np.zeros((self.target_length, kin.shape[1]))
        for i in range(kin.shape[1]):
            f = interp1d(x_kin, kin[:, i], kind='linear', fill_value='extrapolate')
            kin_aligned[:, i] = f(x_target)
        
        return emg_aligned, kin_aligned


class HierarchicalFusionFeatureExtractor(nn.Module):
    def __init__(self, emg_dim=12, kin_dim=63, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emg_proj = nn.Linear(emg_dim, hidden_dim)
        self.kin_proj = nn.Linear(kin_dim, hidden_dim)
        self.hierarchical_fusion = HierarchicalFusionNetwork(
            modalities=["emg", "kin"],
            input_dims={"emg": hidden_dim, "kin": hidden_dim},
            feature_dims=[hidden_dim, hidden_dim, hidden_dim, hidden_dim * 2],
            fusion_strategy="attention",
            dropout=dropout,
            use_skip_connections=True,
        )
    
    def forward(self, emg_seq, kin_seq):
        emg_feat = self.emg_proj(emg_seq)
        kin_feat = self.kin_proj(kin_seq)
        hier_inputs = {"emg": emg_feat, "kin": kin_feat}
        hier_out = self.hierarchical_fusion(hier_inputs)
        hier_features = hier_out["output"]
        if hier_features.ndim == 3:
            hier_features = hier_features.mean(dim=1)
        return hier_features


class SynergyFeatureExtractor:
    def __init__(self, n_synergies=5):
        self.n_synergies = n_synergies
    
    def extract_synergy_features(self, emg):
        features = []
        try:
            emg_nonneg = np.abs(emg) + 1e-10
            n_components = min(self.n_synergies, emg.shape[1])
            nmf = NMF(n_components=n_components, random_state=42, max_iter=500)
            W = nmf.fit_transform(emg_nonneg)
            H = nmf.components_
            features.extend(W.mean(axis=0))
            features.extend(W.std(axis=0))
            features.extend(H.mean(axis=1))
            features.extend(H.std(axis=1))
            for i in range(n_components):
                active_time = (W[:, i] > W[:, i].mean()).sum() / len(W)
                features.append(active_time)
            if n_components > 1:
                corr = np.corrcoef(W.T)
                if not np.isnan(corr).any():
                    off_diag = corr[np.triu_indices(n_components, k=1)]
                    features.extend([np.mean(off_diag), np.std(off_diag)])
                else:
                    features.extend([0, 0])
            else:
                features.extend([0, 0])
            reconstruction = W @ H
            mse = np.mean((emg_nonneg - reconstruction) ** 2)
            features.append(mse)
        except Exception as e:
            n_default = self.n_synergies * 6 + 3
            features = [0] * n_default
        return np.array(features, dtype=np.float32)


# ============================================================================
# 可配置特征提取器（支持消融）
# ============================================================================
class AblationFeatureExtractor:
    """
    支持消融实验的特征提取器
    
    参数:
        ablation: 消融类型 (1-10)
        n_synergies: 肌肉协同性数量
        use_fusion: 是否使用层次化融合
        use_synergy: 是否使用肌肉协同性特征
        use_covariance: 是否使用EMG协方差特征
        use_freq: 是否使用EMG频域特征
        use_nonlinear: 是否使用EMG非线性特征
        use_kin: 是否使用KIN特征
    """
    
    def __init__(self, ablation=1, n_synergies=5, use_fusion=True, use_synergy=True,
                 use_covariance=True, use_freq=True, use_nonlinear=True, use_kin=True):
        self.fs_emg = 1000
        self.fs_kin = 100
        self.ablation = ablation
        self.n_synergies = n_synergies
        self.use_fusion = use_fusion
        self.use_synergy = use_synergy
        self.use_covariance = use_covariance
        self.use_freq = use_freq
        self.use_nonlinear = use_nonlinear
        self.use_kin = use_kin
        
        # 根据消融类型配置
        self._configure_ablation()
        
        # 初始化组件
        self.synergy_extractor = SynergyFeatureExtractor(n_synergies=n_synergies)
        
        if self.use_fusion:
            self.fusion_extractor = HierarchicalFusionFeatureExtractor(
                emg_dim=12, kin_dim=63, hidden_dim=Config.HIDDEN_DIM, dropout=0.3
            ).to(Config.DEVICE)
            self.fusion_extractor.eval()
    
    def _configure_ablation(self):
        """根据消融类型配置特征开关"""
        if self.ablation == 1:  # 完整模型
            pass
        elif self.ablation == 2:  # 去除融合特征
            self.use_fusion = False
        elif self.ablation == 3:  # 去除肌肉协同性
            self.use_synergy = False
        elif self.ablation == 4:  # 去除EMG协方差
            self.use_covariance = False
        elif self.ablation == 5:  # 去除EMG频域
            self.use_freq = False
        elif self.ablation == 6:  # 去除EMG非线性
            self.use_nonlinear = False
        elif self.ablation == 7:  # 去除KIN特征
            self.use_kin = False
        elif self.ablation == 8:  # 仅EMG
            self.use_fusion = False
            self.use_kin = False
        elif self.ablation == 9:  # 仅KIN
            self.use_fusion = False
            self.use_synergy = False
            self.use_covariance = False
            self.use_freq = False
            self.use_nonlinear = False
        elif self.ablation == 10:  # 仅融合
            self.use_synergy = False
            self.use_covariance = False
            self.use_freq = False
            self.use_nonlinear = False
            self.use_kin = False
    
    def extract_emg_time_features(self, emg):
        features = []
        for ch in range(emg.shape[1]):
            x = emg[:, ch]
            features.extend([
                np.mean(np.abs(x)), np.sqrt(np.mean(x**2)), np.var(x), np.std(x),
                np.max(x), np.min(x), np.max(x) - np.min(x),
            ])
            diff = np.diff(x)
            features.extend([
                np.sum(np.abs(diff)) / len(diff), np.mean(np.abs(diff)), np.std(diff),
            ])
            zero_crossings = np.sum(np.abs(np.diff(np.sign(x)))) / 2
            features.append(zero_crossings / len(x))
            slope_sign_changes = np.sum(np.abs(np.diff(np.sign(diff)))) / 2
            features.append(slope_sign_changes / len(diff) if len(diff) > 0 else 0)
        return np.array(features, dtype=np.float32)
    
    def extract_emg_freq_features(self, emg):
        if not self.use_freq:
            return np.array([], dtype=np.float32)
        features = []
        freq_bands = {'low': (0, 50), 'mid_low': (50, 100), 'mid': (100, 200), 
                      'mid_high': (200, 350), 'high': (350, 500)}
        for ch in range(emg.shape[1]):
            x = emg[:, ch]
            freqs, psd = welch(x, fs=self.fs_emg, nperseg=min(256, len(x)))
            if len(psd) > 0 and np.sum(psd) > 0:
                total_power = np.sum(psd)
                for band_name, (low, high) in freq_bands.items():
                    band_mask = (freqs >= low) & (freqs < high)
                    band_power = np.sum(psd[band_mask]) if band_mask.sum() > 0 else 0
                    features.append(band_power / (total_power + 1e-10))
                cumsum = np.cumsum(psd)
                features.append(freqs[np.searchsorted(cumsum, cumsum[-1] / 2)])
                features.append(np.sum(freqs * psd) / (total_power + 1e-10))
                features.append(freqs[np.argmax(psd)])
                features.append(entropy(psd / (total_power + 1e-10) + 1e-10))
            else:
                features.extend([0] * 9)
        return np.array(features, dtype=np.float32)
    
    def extract_emg_covariance_features(self, emg):
        if not self.use_covariance:
            return np.array([], dtype=np.float32)
        features = []
        cov_matrix = np.cov(emg.T)
        diag = np.diag(cov_matrix)
        features.extend([np.mean(diag), np.std(diag), np.max(diag), np.min(diag)])
        n_channels = emg.shape[1]
        if n_channels > 1:
            off_diag = cov_matrix[np.triu_indices(n_channels, k=1)]
            features.extend([np.mean(np.abs(off_diag)), np.std(off_diag), 
                           np.max(off_diag), np.min(off_diag)])
            corr_matrix = np.corrcoef(emg.T)
            if not np.isnan(corr_matrix).any():
                off_diag_corr = corr_matrix[np.triu_indices(n_channels, k=1)]
                features.extend([np.mean(off_diag_corr), np.std(off_diag_corr),
                               np.max(off_diag_corr), np.min(off_diag_corr)])
            else:
                features.extend([0, 0, 0, 0])
        else:
            features.extend([0, 0, 0, 0, 0, 0, 0, 0])
        return np.array(features, dtype=np.float32)
    
    def extract_emg_nonlinear_features(self, emg):
        if not self.use_nonlinear:
            return np.array([], dtype=np.float32)
        features = []
        for ch in range(emg.shape[1]):
            x = emg[:, ch]
            features.append(kurtosis(x))
            features.append(skew(x))
            if len(x) > 10:
                diff = np.diff(x)
                features.append(np.std(diff) / (np.std(x) + 1e-10))
            else:
                features.append(0)
        return np.array(features, dtype=np.float32)
    
    def extract_synergy_features(self, emg):
        if not self.use_synergy:
            return np.array([], dtype=np.float32)
        return self.synergy_extractor.extract_synergy_features(emg)
    
    def extract_fusion_features(self, emg, kin):
        if not self.use_fusion:
            return np.array([], dtype=np.float32)
        with torch.no_grad():
            emg_tensor = torch.FloatTensor(emg).unsqueeze(0).to(Config.DEVICE)
            kin_tensor = torch.FloatTensor(kin).unsqueeze(0).to(Config.DEVICE)
            fusion_feat = self.fusion_extractor(emg_tensor, kin_tensor)
            return fusion_feat.squeeze().cpu().numpy()
    
    def extract_emg_features(self, emg):
        feat_list = []
        feat_list.append(self.extract_emg_time_features(emg))
        freq_feat = self.extract_emg_freq_features(emg)
        if len(freq_feat) > 0:
            feat_list.append(freq_feat)
        cov_feat = self.extract_emg_covariance_features(emg)
        if len(cov_feat) > 0:
            feat_list.append(cov_feat)
        nonlinear_feat = self.extract_emg_nonlinear_features(emg)
        if len(nonlinear_feat) > 0:
            feat_list.append(nonlinear_feat)
        synergy_feat = self.extract_synergy_features(emg)
        if len(synergy_feat) > 0:
            feat_list.append(synergy_feat)
        return np.concatenate(feat_list) if feat_list else np.array([], dtype=np.float32)
    
    def extract_kin_features(self, kin):
        if not self.use_kin:
            return np.array([], dtype=np.float32)
        features = []
        features.extend([np.mean(kin), np.std(kin), np.max(kin), np.min(kin), 
                        np.max(kin) - np.min(kin)])
        vel = np.diff(kin, axis=0)
        vel_mag = np.sqrt(np.sum(vel**2, axis=1))
        if len(vel_mag) > 0:
            features.extend([np.mean(vel_mag), np.std(vel_mag), np.max(vel_mag)])
        else:
            features.extend([0, 0, 0])
        if len(vel) > 1:
            acc = np.diff(vel, axis=0)
            acc_mag = np.sqrt(np.sum(acc**2, axis=1))
            if len(acc_mag) > 0:
                features.extend([np.mean(acc_mag), np.std(acc_mag)])
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0])
        if len(kin) > 3:
            vel = np.gradient(kin, axis=0)
            acc = np.gradient(vel, axis=0)
            jerk = np.gradient(acc, axis=0)
            jerk_mag = np.sqrt(np.sum(jerk**2, axis=1))
            if len(jerk_mag) > 0:
                features.append(1 / (1 + np.sum(jerk_mag**2) * 0.01))
            else:
                features.append(0)
        else:
            features.append(0)
        return np.array(features, dtype=np.float32)
    
    def extract_all_features(self, emg, kin):
        """提取所有特征（根据消融配置）"""
        feat_list = []
        
        # EMG特征
        emg_feat = self.extract_emg_features(emg)
        if len(emg_feat) > 0:
            feat_list.append(emg_feat)
        
        # KIN特征
        kin_feat = self.extract_kin_features(kin)
        if len(kin_feat) > 0:
            feat_list.append(kin_feat)
        
        # 融合特征
        fusion_feat = self.extract_fusion_features(emg, kin)
        if len(fusion_feat) > 0:
            feat_list.append(fusion_feat)
        
        return np.concatenate(feat_list) if feat_list else np.array([], dtype=np.float32)


# ============================================================================
# 受试者级特征聚合
# ============================================================================
def extract_subject_features(sample_features):
    if sample_features.shape[1] == 0:
        return np.array([], dtype=np.float32)
    subject_feat = []
    for i in range(sample_features.shape[1]):
        feat = sample_features[:, i]
        subject_feat.extend([
            np.mean(feat), np.std(feat), np.min(feat), np.max(feat), np.median(feat),
            np.percentile(feat, 25), np.percentile(feat, 75),
            kurtosis(feat) if len(feat) > 3 else 0, skew(feat) if len(feat) > 3 else 0,
        ])
    return np.array(subject_feat, dtype=np.float32)


# ============================================================================
# 数据加载
# ============================================================================
def load_and_extract_features(ablation=1, n_synergies=5):
    print("="*60)
    print(f"加载数据并提取特征 (消融类型={ablation}, n_synergies={n_synergies})...")
    print("="*60)
    
    loader = SCIDataLoader(data_root=Config.DATA_DIR)
    raw_samples = loader.load_all_samples(group="stroke")
    print(f"原始样本: {len(raw_samples)}")
    
    dtw_aligner = DTWAligner(target_length=Config.TARGET_LENGTH)
    feature_extractor = AblationFeatureExtractor(
        ablation=ablation, 
        n_synergies=n_synergies
    )
    
    subject_samples = defaultdict(list)
    subject_fma = {}
    
    print("\n特征提取中...")
    for sample in raw_samples:
        if sample.emg_data.size == 0 or sample.kin_data.size == 0:
            continue
        if sample.fma_score is None:
            continue
        
        try:
            emg_aligned, kin_aligned = dtw_aligner.align(
                sample.emg_data, 
                sample.kin_data[:, :63] if sample.kin_data.shape[1] >= 63 else sample.kin_data
            )
            
            feat = feature_extractor.extract_all_features(emg_aligned, kin_aligned)
            
            if len(feat) == 0 or np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
                continue
            
            sid = f"S{sample.subject_id}"
            subject_samples[sid].append(feat)
            subject_fma[sid] = sample.fma_score
            
        except Exception as e:
            continue
    
    X, y, sids = [], [], []
    
    print("\n聚合到受试者级别...")
    for sid, feats in subject_samples.items():
        if len(feats) > 0:
            feats_array = np.array(feats)
            subj_feat = extract_subject_features(feats_array)
            if len(subj_feat) > 0:
                X.append(subj_feat)
                y.append(subject_fma[sid])
                sids.append(sid)
    
    X = np.array(X, dtype=np.float32) if X else np.array([]).reshape(0, 0)
    y = np.array(y, dtype=np.float32) if y else np.array([])
    
    print(f"\n受试者数量: {len(sids)}")
    if len(X) > 0:
        print(f"受试者级特征维度: {X.shape[1]}")
    print(f"FMA分布: {dict(pd.Series(y).value_counts().sort_index())}")
    
    return X, y, sids


# ============================================================================
# 特征选择
# ============================================================================
def select_features(X, y, method='elastic_net'):
    """特征选择"""
    if method == 'none':
        return X, np.arange(X.shape[1])
    
    if method == 'elastic_net':
        elastic = ElasticNetCV(cv=5, random_state=42, max_iter=10000, l1_ratio=0.5)
        elastic.fit(X, y)
        selected_idx = np.where(np.abs(elastic.coef_) > 1e-5)[0]
        if len(selected_idx) == 0:
            selected_idx = np.arange(X.shape[1])
    
    elif method == 'f_test':
        n_select = min(50, X.shape[1] - 1)
        selector = SelectKBest(f_regression, k=n_select)
        X_selected = selector.fit_transform(X, y)
        selected_idx = selector.get_support(indices=True)
        return X_selected, selected_idx
    
    else:
        selected_idx = np.arange(X.shape[1])
    
    return X[:, selected_idx], selected_idx


# ============================================================================
# 降维
# ============================================================================
def reduce_dimension(X, method='pca', n_components=15):
    """降维"""
    if method == 'none' or X.shape[1] <= n_components:
        return X, 1.0
    
    pca = PCA(n_components=min(n_components, X.shape[1] - 1))
    X_reduced = pca.fit_transform(X)
    explained_var = np.sum(pca.explained_variance_ratio_)
    
    return X_reduced, explained_var


# ============================================================================
# 模型训练
# ============================================================================
def train_model(X_train, y_train, X_test, model_type='ensemble'):
    """训练模型并预测"""
    
    if model_type == 'svr':
        model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        model.fit(X_train, y_train)
        return model.predict(X_test)[0]
    
    elif model_type == 'knn':
        model = KNeighborsRegressor(n_neighbors=3)
        model.fit(X_train, y_train)
        return model.predict(X_test)[0]
    
    elif model_type == 'gb':
        model = GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)
        model.fit(X_train, y_train)
        return model.predict(X_test)[0]
    
    elif model_type == 'ensemble':
        svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        knn = KNeighborsRegressor(n_neighbors=3)
        gb = GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)
        
        svr.fit(X_train, y_train)
        knn.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        
        w_svr, w_knn, w_gb = 0.24, 0.22, 0.54
        pred = (w_svr * svr.predict(X_test)[0] + 
                w_knn * knn.predict(X_test)[0] + 
                w_gb * gb.predict(X_test)[0])
        return pred
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# LOSOCV评估
# ============================================================================
def evaluate_losocv(X, y, sids, model_type='ensemble'):
    """留一受试者交叉验证"""
    loo = LeaveOneOut()
    all_preds = []
    all_labels = []
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        pred = train_model(X_train, y_train, X_test, model_type)
        all_preds.append(pred)
        all_labels.append(y_test[0])
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算指标
    mae = np.mean(np.abs(all_preds - all_labels))
    simple_round = np.array([
        Config.FMA_VALUES[np.argmin(np.abs(np.array(Config.FMA_VALUES) - p))] 
        for p in all_preds
    ])
    acc = np.mean(simple_round == all_labels)
    acc_pm1 = np.mean(np.abs(simple_round - all_labels) <= 1)
    
    return mae, acc, acc_pm1, all_preds, all_labels


# ============================================================================
# 单次消融实验
# ============================================================================
def run_single_ablation(ablation=1, n_synergies=5, feature_selection='elastic_net',
                        dim_reduction='pca', model_type='ensemble'):
    """运行单次消融实验"""
    
    print("\n" + "="*60)
    print(f"消融实验: {ABLATION_NAMES.get(ablation, f'自定义配置')}")
    print(f"参数: n_synergies={n_synergies}, feature_selection={feature_selection}, "
          f"dim_reduction={dim_reduction}, model={model_type}")
    print("="*60)
    
    # 加载数据
    X, y, sids = load_and_extract_features(ablation=ablation, n_synergies=n_synergies)
    
    if len(X) == 0 or X.shape[1] == 0:
        print("错误: 特征提取失败，无有效特征")
        return None
    
    # 标准化
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 特征选择
    print(f"\n特征选择: {feature_selection}")
    X_selected, selected_idx = select_features(X_scaled, y, method=feature_selection)
    print(f"选择后特征数: {X_selected.shape[1]}")
    
    # 降维
    print(f"\n降维: {dim_reduction}")
    X_reduced, explained_var = reduce_dimension(X_selected, method=dim_reduction)
    print(f"最终特征维度: {X_reduced.shape[1]}, 解释方差: {explained_var:.2%}")
    
    # LOSOCV评估
    print(f"\nLOSOCV训练 (模型: {model_type})...")
    mae, acc, acc_pm1, preds, labels = evaluate_losocv(
        X_reduced, y, sids, model_type=model_type
    )
    
    print(f"\n结果: MAE={mae:.2f}, Acc@1={acc:.1%}, Acc@±1={acc_pm1:.1%}")
    
    return {
        'ablation': ablation,
        'ablation_name': ABLATION_NAMES.get(ablation, '自定义'),
        'n_synergies': n_synergies,
        'feature_selection': feature_selection,
        'dim_reduction': dim_reduction,
        'model_type': model_type,
        'n_features': X.shape[1],
        'n_selected': X_selected.shape[1],
        'n_final': X_reduced.shape[1],
        'explained_var': explained_var,
        'mae': mae,
        'acc': acc,
        'acc_pm1': acc_pm1,
        'preds': preds,
        'labels': labels,
        'sids': sids,
    }


# ============================================================================
# 批量消融实验
# ============================================================================
def run_all_ablations():
    """运行所有消融实验"""
    print("\n" + "="*60)
    print("开始批量消融实验")
    print("="*60)
    
    results = []
    
    # 1. 特征消融 (ablation 1-10)
    print("\n" + "="*60)
    print("第一部分: 特征消融实验")
    print("="*60)
    
    for ablation in range(1, 11):
        result = run_single_ablation(
            ablation=ablation,
            n_synergies=5,
            feature_selection='elastic_net',
            dim_reduction='pca',
            model_type='ensemble'
        )
        if result:
            results.append(result)
    
    # 2. n_synergies消融
    print("\n" + "="*60)
    print("第二部分: n_synergies参数消融")
    print("="*60)
    
    for n_syn in [3, 4, 5, 6, 7]:
        result = run_single_ablation(
            ablation=1,
            n_synergies=n_syn,
            feature_selection='elastic_net',
            dim_reduction='pca',
            model_type='ensemble'
        )
        if result:
            result['ablation_name'] = f'n_synergies={n_syn}'
            results.append(result)
    
    # 3. 特征选择方法消融
    print("\n" + "="*60)
    print("第三部分: 特征选择方法消融")
    print("="*60)
    
    for fs_method in ['elastic_net', 'f_test', 'none']:
        result = run_single_ablation(
            ablation=1,
            n_synergies=5,
            feature_selection=fs_method,
            dim_reduction='pca',
            model_type='ensemble'
        )
        if result:
            result['ablation_name'] = f'特征选择={fs_method}'
            results.append(result)
    
    # 4. 降维方法消融
    print("\n" + "="*60)
    print("第四部分: 降维方法消融")
    print("="*60)
    
    for dr_method in ['pca', 'none']:
        result = run_single_ablation(
            ablation=1,
            n_synergies=5,
            feature_selection='elastic_net',
            dim_reduction=dr_method,
            model_type='ensemble'
        )
        if result:
            result['ablation_name'] = f'降维={dr_method}'
            results.append(result)
    
    # 5. 模型消融
    print("\n" + "="*60)
    print("第五部分: 模型消融")
    print("="*60)
    
    for model in ['svr', 'knn', 'gb', 'ensemble']:
        result = run_single_ablation(
            ablation=1,
            n_synergies=5,
            feature_selection='elastic_net',
            dim_reduction='pca',
            model_type=model
        )
        if result:
            result['ablation_name'] = f'模型={model}'
            results.append(result)
    
    return results


# ============================================================================
# 保存结果
# ============================================================================
def save_results(results, output_dir):
    """保存消融实验结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 汇总表
    summary_data = []
    for r in results:
        summary_data.append({
            '实验名称': r['ablation_name'],
            'ablation': r['ablation'],
            'n_synergies': r['n_synergies'],
            '特征选择': r['feature_selection'],
            '降维方法': r['dim_reduction'],
            '模型': r['model_type'],
            '原始特征数': r['n_features'],
            '选择后特征数': r['n_selected'],
            '最终特征数': r['n_final'],
            '解释方差': f"{r['explained_var']:.2%}",
            'MAE': f"{r['mae']:.2f}",
            'Acc@1': f"{r['acc']:.1%}",
            'Acc@±1': f"{r['acc_pm1']:.1%}",
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(f'{output_dir}/ablation_summary.csv', index=False, encoding='utf-8-sig')
    
    # 详细结果
    with open(f'{output_dir}/ablation_details.json', 'w', encoding='utf-8') as f:
        # 转换为可序列化格式
        serializable_results = []
        for r in results:
            sr = {}
            for k, v in r.items():
                if k in ['preds', 'labels']:
                    sr[k] = [float(x) for x in v.tolist()] if hasattr(v, 'tolist') else []
                elif k == 'sids':
                    sr[k] = list(v) if isinstance(v, (list, np.ndarray)) else []
                elif isinstance(v, (np.floating, np.integer)):
                    sr[k] = float(v)
                elif isinstance(v, np.ndarray):
                    sr[k] = v.tolist()
                else:
                    sr[k] = v
            serializable_results.append(sr)
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存: {output_dir}/")
    
    return df_summary


# ============================================================================
# 打印汇总报告
# ============================================================================
def print_summary_report(results):
    """打印汇总报告"""
    print("\n" + "="*80)
    print("消融实验汇总报告")
    print("="*80)
    
    # 按MAE排序
    sorted_results = sorted(results, key=lambda x: x['mae'])
    
    print(f"\n{'实验名称':<30} {'MAE':<8} {'Acc@1':<8} {'Acc@±1':<8}")
    print("-" * 60)
    
    for r in sorted_results:
        print(f"{r['ablation_name']:<30} {r['mae']:<8.2f} {r['acc']:<8.1%} {r['acc_pm1']:<8.1%}")
    
    # 最佳结果
    best = sorted_results[0]
    print("\n" + "="*80)
    print("最佳配置")
    print("="*80)
    print(f"实验名称: {best['ablation_name']}")
    print(f"MAE: {best['mae']:.2f}")
    print(f"Acc@1: {best['acc']:.1%}")
    print(f"Acc@±1: {best['acc_pm1']:.1%}")
    print(f"特征数: {best['n_final']}")
    
    # 关键发现
    print("\n" + "="*80)
    print("关键发现")
    print("="*80)
    
    # 特征消融分析
    feature_results = [r for r in results if r['ablation'] >= 1 and r['ablation'] <= 10]
    if feature_results:
        baseline = [r for r in feature_results if r['ablation'] == 1][0]
        print("\n1. 特征消融分析:")
        for r in feature_results:
            if r['ablation'] != 1:
                delta_mae = r['mae'] - baseline['mae']
                impact = "恶化" if delta_mae > 0.1 else "轻微影响" if abs(delta_mae) <= 0.1 else "改善"
                print(f"   - {r['ablation_name']}: MAE变化 {delta_mae:+.2f} ({impact})")
    
    # 模型消融分析
    model_results = [r for r in results if '模型=' in r['ablation_name']]
    if model_results:
        print("\n2. 模型消融分析:")
        for r in sorted(model_results, key=lambda x: x['mae']):
            print(f"   - {r['ablation_name']}: MAE={r['mae']:.2f}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='HybridSCIMFP 消融实验')
    parser.add_argument('--ablation', type=int, default=None, 
                       help='消融类型 (1-10), 不指定则运行全部')
    parser.add_argument('--n_synergies', type=int, default=5,
                       help='肌肉协同性数量 (3-7)')
    parser.add_argument('--feature_selection', type=str, default='elastic_net',
                       choices=['elastic_net', 'f_test', 'none'],
                       help='特征选择方法')
    parser.add_argument('--dim_reduction', type=str, default='pca',
                       choices=['pca', 'none'],
                       help='降维方法')
    parser.add_argument('--model', type=str, default='ensemble',
                       choices=['svr', 'knn', 'gb', 'ensemble'],
                       help='预测模型')
    parser.add_argument('--output_dir', type=str, default=Config.OUTPUT_DIR,
                       help='输出目录')
    
    args = parser.parse_args()
    
    print("="*80)
    print("HybridSCIMFP - 消融实验 (Ablation Study)")
    print("="*80)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输出目录: {args.output_dir}")
    
    if args.ablation is not None:
        # 单次消融实验
        result = run_single_ablation(
            ablation=args.ablation,
            n_synergies=args.n_synergies,
            feature_selection=args.feature_selection,
            dim_reduction=args.dim_reduction,
            model_type=args.model
        )
        results = [result] if result else []
    else:
        # 批量消融实验
        results = run_all_ablations()
    
    if results:
        # 保存结果
        df_summary = save_results(results, args.output_dir)
        
        # 打印汇总报告
        print_summary_report(results)
        
        # 打印表格
        print("\n" + "="*80)
        print("结果汇总表")
        print("="*80)
        print(df_summary.to_string(index=False))
    
    print("\n" + "="*80)
    print("消融实验完成")
    print("="*80)
    
    return results


if __name__ == '__main__':
    results = main()

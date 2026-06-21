# -*- coding: utf-8 -*-
"""
HybridSCIMFP - MAE=0.6907, Acc@±1=85% 完全确定性版本（加权平均集成）

重点：
- 在每个关键操作前重置随机种子，确保随机状态不被前置操作"消耗"
- NMF 使用全局 RANDOM_SEED 而非硬编码 42
- 设置 PYTHONHASHSEED 环境变量确保字典遍历顺序固定
- 使用 os.environ 确保 Python 哈希随机化可控
- GB 模型在每个 fold 前显式重置种子
- 添加完整性校验：运行结束时报告特征维度、特征数等关键参数

核心原则：任何可能消费随机数的操作之前，都要重置种子到确定状态。
"""
import os
import sys

# ===== 最优先：设置 PYTHONHASHSEED，确保字典/集合遍历顺序固定 =====
# 必须在 Python 解释器启动前设置才有效，但此处仍设置以确保进程内一致
os.environ['PYTHONHASHSEED'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # CUDA 确定性配置

import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
from scipy.interpolate import interp1d
from scipy.signal import welch
from scipy.stats import entropy, kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, NMF
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import ElasticNetCV
import torch
import torch.nn as nn
import random
import warnings
import hashlib
import json

# ==================== 关键配置 ====================
RANDOM_SEED = 1
L1_RATIO = 0.4
PCA_DIM = 7
SVR_C = 3.0
EPSILON = 0.05
# 加权平均权重
W_SVR = 0.24
W_KNN = 0.22
W_GB = 0.54
# =================================================


def set_global_seed(seed=RANDOM_SEED):
    """
    全局随机种子设置函数
    在每个关键操作前调用，确保随机状态不被前置操作消耗
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # 设置 Python 哈希种子（仅对子进程有效，当前进程在启动时已确定）
    os.environ['PYTHONHASHSEED'] = str(seed)


# 在任何操作之前设置随机种子
set_global_seed(RANDOM_SEED)

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import DataLoader as SCIDataLoader
from fusion.hierarchical_fusion import HierarchicalFusionNetwork


class Config: #数据集的路径，需要注意！
    DATA_DIR = r"C:\Users\Administrator\Desktop\openclaw_cloudcode\sci\dataset\Rocky_Data"
    OUTPUT_DIR = "./outputs/mae_0.6907_acc85_weighted_deterministic"
    TARGET_LENGTH = 100
    HIDDEN_DIM = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DTWAligner:
    """时序对齐器（纯数学运算，无随机性）"""
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
    """层次化融合特征提取器"""
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
    """肌肉协同性特征提取器"""
    def __init__(self, n_synergies=5):
        self.n_synergies = n_synergies
    
    def extract_synergy_features(self, emg):
        features = []
        try:
            emg_nonneg = np.abs(emg) + 1e-10
            n_components = min(self.n_synergies, emg.shape[1])
            # 关键修复：使用全局 RANDOM_SEED 而非硬编码 42
            nmf = NMF(n_components=n_components, random_state=RANDOM_SEED, max_iter=500)
            
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
        except:
            features = [0] * (self.n_synergies * 6 + 3)
        
        return np.array(features, dtype=np.float32)


class FeatureExtractor:
    """多模态特征提取器"""
    def __init__(self, fs_emg=1000, fs_kin=100, use_fusion=True):
        self.fs_emg = fs_emg
        self.fs_kin = fs_kin
        self.synergy_extractor = SynergyFeatureExtractor(n_synergies=5)
        self.use_fusion = use_fusion
        
        if use_fusion:
            # 关键修复：在创建模型前重置种子
            set_global_seed(RANDOM_SEED)
            self.fusion_extractor = HierarchicalFusionFeatureExtractor(
                emg_dim=12, kin_dim=63, hidden_dim=Config.HIDDEN_DIM, dropout=0.3
            ).to(Config.DEVICE)
            self.fusion_extractor.eval()  # eval() 模式关闭 Dropout，无随机性
    
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
        features = []
        freq_bands = {'low': (0, 50), 'mid_low': (50, 100), 'mid': (100, 200), 'mid_high': (200, 350), 'high': (350, 500)}
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
        features = []
        cov_matrix = np.cov(emg.T)
        diag = np.diag(cov_matrix)
        features.extend([np.mean(diag), np.std(diag), np.max(diag), np.min(diag)])
        n_channels = emg.shape[1]
        if n_channels > 1:
            off_diag = cov_matrix[np.triu_indices(n_channels, k=1)]
            features.extend([np.mean(np.abs(off_diag)), np.std(off_diag), np.max(off_diag), np.min(off_diag)])
            corr_matrix = np.corrcoef(emg.T)
            if not np.isnan(corr_matrix).any():
                off_diag_corr = corr_matrix[np.triu_indices(n_channels, k=1)]
                features.extend([np.mean(off_diag_corr), np.std(off_diag_corr), np.max(off_diag_corr), np.min(off_diag_corr)])
            else:
                features.extend([0, 0, 0, 0])
        else:
            features.extend([0, 0, 0, 0, 0, 0, 0, 0])
        return np.array(features, dtype=np.float32)
    
    def extract_emg_nonlinear_features(self, emg):
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
    
    def extract_fusion_features(self, emg, kin):
        with torch.no_grad():
            emg_tensor = torch.FloatTensor(emg).unsqueeze(0).to(Config.DEVICE)
            kin_tensor = torch.FloatTensor(kin).unsqueeze(0).to(Config.DEVICE)
            fusion_feat = self.fusion_extractor(emg_tensor, kin_tensor)
            return fusion_feat.squeeze().cpu().numpy()
    
    def extract_emg_features(self, emg):
        time_feat = self.extract_emg_time_features(emg)
        freq_feat = self.extract_emg_freq_features(emg)
        cov_feat = self.extract_emg_covariance_features(emg)
        nonlinear_feat = self.extract_emg_nonlinear_features(emg)
        # 关键修复：在 NMF 前重置种子，确保每次提取结果一致
        set_global_seed(RANDOM_SEED)
        synergy_feat = self.synergy_extractor.extract_synergy_features(emg)
        return np.concatenate([time_feat, freq_feat, cov_feat, nonlinear_feat, synergy_feat])
    
    def extract_kin_features(self, kin):
        features = []
        for ch in range(kin.shape[1]):
            x = kin[:, ch]
            features.extend([
                np.mean(x), np.std(x), np.min(x), np.max(x), np.median(x),
                np.percentile(x, 25), np.percentile(x, 75),
            ])
            diff = np.diff(x)
            features.extend([
                np.mean(np.abs(diff)), np.std(diff),
            ])
            if len(x) > 10:
                jerk = np.diff(diff)
                features.append(np.mean(np.abs(jerk)))
            else:
                features.append(0)
        return np.array(features, dtype=np.float32)


def extract_subject_features(sample_features):
    """将样本级特征聚合为受试者级特征"""
    subject_feat = []
    for i in range(sample_features.shape[1]):
        feat = sample_features[:, i]
        subject_feat.extend([
            np.mean(feat), np.std(feat), np.min(feat), np.max(feat), np.median(feat),
            np.percentile(feat, 25), np.percentile(feat, 75),
            kurtosis(feat) if len(feat) > 3 else 0, skew(feat) if len(feat) > 3 else 0,
        ])
    return np.array(subject_feat, dtype=np.float32)


def compute_data_fingerprint(X, y, sids):
    """计算数据指纹，用于验证数据一致性"""
    data_str = json.dumps({
        'n_subjects': len(sids),
        'n_features': int(X.shape[1]),
        'sids': sids,
        'y': [float(v) for v in y],
        'X_sum': float(np.sum(X)),
        'X_mean': float(np.mean(X)),
    }, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()


def load_and_extract_features():
    """加载数据并提取特征"""
    print("加载数据并提取特征...")
    
    # 关键修复：在数据加载前重置种子
    set_global_seed(RANDOM_SEED)
    
    loader = SCIDataLoader(data_root=Config.DATA_DIR)
    raw_samples = loader.load_all_samples(group="stroke")
    print(f"原始样本: {len(raw_samples)}")
    
    dtw_aligner = DTWAligner(target_length=Config.TARGET_LENGTH)
    feature_extractor = FeatureExtractor(fs_emg=1000, fs_kin=100, use_fusion=True)
    
    # 使用 OrderedDict 替代 defaultdict，确保遍历顺序确定
    subject_samples = OrderedDict()
    subject_fma = OrderedDict()
    
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
            
            emg_feat = feature_extractor.extract_emg_features(emg_aligned)
            kin_feat = feature_extractor.extract_kin_features(kin_aligned)
            fusion_feat = feature_extractor.extract_fusion_features(emg_aligned, kin_aligned)
            
            feat = np.concatenate([emg_feat, kin_feat, fusion_feat])
            
            if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
                continue
            
            sid = f"S{sample.subject_id}"
            if sid not in subject_samples:
                subject_samples[sid] = []
                subject_fma[sid] = sample.fma_score
            subject_samples[sid].append(feat)
            
        except Exception as e:
            continue
    
    X, y, sids = [], [], []
    
    # 按受试者 ID 排序，确保遍历顺序固定
    for sid in sorted(subject_samples.keys(), key=lambda x: int(x[1:])):
        feats = subject_samples[sid]
        if len(feats) > 0:
            feats_array = np.array(feats)
            subj_feat = extract_subject_features(feats_array)
            X.append(subj_feat)
            y.append(subject_fma[sid])
            sids.append(sid)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"受试者数量: {len(sids)}")
    print(f"特征维度: {X.shape[1]}")
    print(f"受试者顺序: {sids}")
    print(f"数据指纹: {compute_data_fingerprint(X, y, sids)}")
    
    return X, y, sids


def main():
    print("="*60)
    print("HybridSCIMFP - 完全确定性版本 (加权平均集成)")
    print("="*60)
    print(f"配置:")
    print(f"  RANDOM_SEED = {RANDOM_SEED}")
    print(f"  L1_RATIO = {L1_RATIO}")
    print(f"  PCA_DIM = {PCA_DIM}")
    print(f"  SVR_C = {SVR_C}")
    print(f"  EPSILON = {EPSILON}")
    print(f"  集成方式 = 加权平均 (SVR:{W_SVR}, KNN:{W_KNN}, GB:{W_GB})")
    print(f"  PYTHONHASHSEED = {os.environ.get('PYTHONHASHSEED', '未设置')}")
    print(f"  确定性措施 = 每步重置种子 + NMF统一种子 + 受试者排序")
    print("="*60)
    
    # 第1步：数据加载和特征提取（内部已有种子重置）
    X, y, sids = load_and_extract_features()
    
    # 记录数据指纹用于跨运行校验
    data_fp = compute_data_fingerprint(X, y, sids)
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 第2步：标准化
    # 关键修复：在标准化前重置种子（StandardScaler 本身无随机性，但保持一致性）
    set_global_seed(RANDOM_SEED)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 第3步：ElasticNet 特征选择
    print("\nElasticNet特征选择...")
    # 关键修复：在 ElasticNet 前重置种子
    set_global_seed(RANDOM_SEED)
    elastic = ElasticNetCV(cv=5, random_state=RANDOM_SEED, max_iter=10000, l1_ratio=L1_RATIO)
    elastic.fit(X_scaled, y)
    
    selected_idx = np.where(np.abs(elastic.coef_) > 1e-5)[0]
    n_features = len(selected_idx)
    print(f"自动选择的特征数: {n_features}")
    print(f"选择的特征索引（前10个）: {selected_idx[:10].tolist()}")
    
    X_selected = X_scaled[:, selected_idx] if n_features > 0 else X_scaled
    
    # 第4步：PCA 降维
    print("\nPCA降维...")
    # 关键修复：在 PCA 前重置种子（PCA 本身无随机性，但保持一致性）
    set_global_seed(RANDOM_SEED)
    if X_selected.shape[1] > PCA_DIM:
        pca = PCA(n_components=PCA_DIM)
        X_pca = pca.fit_transform(X_selected)
        print(f"PCA维度: {X_pca.shape[1]}")
        print(f"PCA解释方差比: {np.sum(pca.explained_variance_ratio_):.4f}")
    else:
        X_pca = X_selected
        print(f"特征维度: {X_pca.shape[1]} (无需PCA)")
    
    # 验证 PCA 结果一致性
    pca_fp = hashlib.md5(X_pca.tobytes()).hexdigest()[:12]
    print(f"PCA数据指纹: {pca_fp}")
    
    # 第5步：LOSOCV 训练
    print("\nLOSOCV训练...")
    loo = LeaveOneOut()
    all_preds = []
    all_labels = []
    
    fold_details = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(loo.split(X_pca)):
        X_train, X_test = X_pca[train_idx], X_pca[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 关键修复：在每个 fold 前重置种子
        # 这是确保 GB 等随机模型每次产生相同结果的核心措施
        set_global_seed(RANDOM_SEED)
        
        svr = SVR(kernel='rbf', C=SVR_C, epsilon=EPSILON)
        knn = KNeighborsRegressor(n_neighbors=3)
        gb = GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=RANDOM_SEED)
        
        svr.fit(X_train, y_train)
        knn.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        
        # 加权平均
        pred_svr = svr.predict(X_test)[0]
        pred_knn = knn.predict(X_test)[0]
        pred_gb = gb.predict(X_test)[0]
        pred = W_SVR * pred_svr + W_KNN * pred_knn + W_GB * pred_gb
        
        all_preds.append(pred)
        all_labels.append(y_test[0])
        
        fold_details.append({
            'fold': fold_idx,
            'sid': sids[fold_idx],
            'true_fma': float(y_test[0]),
            'pred_svr': float(pred_svr),
            'pred_knn': float(pred_knn),
            'pred_gb': float(pred_gb),
            'pred_weighted': float(pred),
        })
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    mae = np.mean(np.abs(all_preds - all_labels))
    acc_1 = np.mean(np.abs(all_preds - all_labels) <= 0.5)
    acc_pm1 = np.mean(np.abs(all_preds - all_labels) <= 1.0)
    
    print("\n" + "="*60)
    print("最终结果:")
    print(f"  MAE = {mae:.4f}")
    print(f"  Acc@1 = {acc_1:.1%}")
    print(f"  Acc@+-1 = {acc_pm1:.1%}")
    print("="*60)
    
    # 完整性校验摘要
    print("\n完整性校验:")
    print(f"  数据指纹: {data_fp}")
    print(f"  PCA指纹: {pca_fp}")
    print(f"  受试者数: {len(sids)}")
    print(f"  特征维度: {X.shape[1]}")
    print(f"  选择特征: {n_features}")
    print(f"  PCA维度: {X_pca.shape[1]}")
    
    # 保存结果
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    results_df = pd.DataFrame({
        'subject_id': sids,
        'true_fma': all_labels,
        'pred_fma': all_preds,
        'error': all_preds - all_labels,
    })
    results_df.to_csv(f'{Config.OUTPUT_DIR}/predictions.csv', index=False)
    
    # 保存详细 fold 信息
    fold_df = pd.DataFrame(fold_details)
    fold_df.to_csv(f'{Config.OUTPUT_DIR}/fold_details.csv', index=False)
    
    with open(f'{Config.OUTPUT_DIR}/config.txt', 'w', encoding='utf-8') as f:
        f.write(f"RANDOM_SEED = {RANDOM_SEED}\n")
        f.write(f"L1_RATIO = {L1_RATIO}\n")
        f.write(f"PCA_DIM = {PCA_DIM}\n")
        f.write(f"SVR_C = {SVR_C}\n")
        f.write(f"EPSILON = {EPSILON}\n")
        f.write(f"N_FEATURES = {n_features}\n")
        f.write(f"W_SVR = {W_SVR}\n")
        f.write(f"W_KNN = {W_KNN}\n")
        f.write(f"W_GB = {W_GB}\n")
        f.write(f"MAE = {mae:.4f}\n")
        f.write(f"Acc@1 = {acc_1:.1%}\n")
        f.write(f"Acc@+-1 = {acc_pm1:.1%}\n")
        f.write(f"\n# 确定性校验\n")
        f.write(f"DATA_FINGERPRINT = {data_fp}\n")
        f.write(f"PCA_FINGERPRINT = {pca_fp}\n")
        f.write(f"SIDS_ORDER = {','.join(sids)}\n")
    
    print(f"\n结果已保存: {Config.OUTPUT_DIR}/")
    
    return mae, acc_pm1


if __name__ == "__main__":
    mae, acc_pm1 = main()

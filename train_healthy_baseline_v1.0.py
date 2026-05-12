# -*- coding: utf-8 -*-
"""
策略2：参考基准方法 - train_healthy_baseline_v1.0.py

核心思想：
- 使用健康受试者的特征统计量作为参考基准
- 中风患者的特征相对于基准的偏差反映其FMA等级
- 偏差越大，FMA越低

目标：MAE<1.40, Acc@1尽可能高
"""
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_regression
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import DataLoader as SCIDataLoader


class Config:
    DATA_DIR = r"C:\Users\Administrator\Desktop\openclaw_cloudcode\sci\dataset\Rocky_Data"
    OUTPUT_DIR = "./outputs/healthy_baseline_v1.0"
    TARGET_LENGTH = 100
    FMA_VALUES = [14, 16, 17, 18, 19, 20]


class DTWAligner:
    def __init__(self, target_length=100):
        self.target_length = target_length
    
    def align(self, emg, kin):
        T_emg, T_kin = len(emg), len(kin)
        x_target = np.linspace(0, 1, self.target_length)
        
        emg_aligned = np.zeros((self.target_length, emg.shape[1]))
        for i in range(emg.shape[1]):
            f = interp1d(np.linspace(0, 1, T_emg), emg[:, i], kind='linear', fill_value='extrapolate')
            emg_aligned[:, i] = f(x_target)
        
        kin_aligned = np.zeros((self.target_length, kin.shape[1]))
        for i in range(kin.shape[1]):
            f = interp1d(np.linspace(0, 1, T_kin), kin[:, i], kind='linear', fill_value='extrapolate')
            kin_aligned[:, i] = f(x_target)
        
        return emg_aligned, kin_aligned


class FeatureExtractor:
    def __init__(self, fs_emg=1000, fs_kin=100):
        self.fs_emg = fs_emg
        self.fs_kin = fs_kin
    
    def extract_emg_features(self, emg):
        features = []
        for ch in range(emg.shape[1]):
            x = emg[:, ch]
            features.extend([
                np.mean(np.abs(x)), np.sqrt(np.mean(x**2)), np.var(x),
                np.std(x), np.max(x), np.min(x),
            ])
            diff = np.diff(x)
            features.extend([np.sum(np.abs(diff)) / len(diff) if len(diff) > 0 else 0])
        
        for ch in range(emg.shape[1]):
            x = emg[:, ch]
            freqs, psd = welch(x, fs=self.fs_emg, nperseg=min(256, len(x)))
            if len(psd) > 0 and np.sum(psd) > 0:
                features.append(np.sum(freqs * psd) / (np.sum(psd) + 1e-10))
            else:
                features.append(0)
        
        return np.array(features, dtype=np.float32)
    
    def extract_kin_features(self, kin):
        features = []
        vel = np.diff(kin, axis=0)
        vel_mag = np.sqrt(np.sum(vel**2, axis=1))
        features.extend([
            np.mean(vel_mag) if len(vel_mag) > 0 else 0,
            np.std(vel_mag) if len(vel_mag) > 0 else 0,
            np.max(vel_mag) if len(vel_mag) > 0 else 0,
        ])
        
        if len(kin) > 3:
            vel = np.gradient(kin, axis=0)
            acc = np.gradient(vel, axis=0)
            jerk = np.gradient(acc, axis=0)
            jerk_mag = np.sqrt(np.sum(jerk**2, axis=1))
            features.append(1 / (1 + np.sum(jerk_mag) / len(jerk_mag)))
        else:
            features.append(0)
        
        features.extend([np.mean(kin), np.std(kin), np.max(kin) - np.min(kin)])
        return np.array(features, dtype=np.float32)


def extract_subject_features(sample_features):
    subject_feat = []
    for i in range(sample_features.shape[1]):
        feat = sample_features[:, i]
        subject_feat.extend([
            np.mean(feat), np.std(feat), np.min(feat), np.max(feat),
            np.median(feat), np.percentile(feat, 25), np.percentile(feat, 75),
        ])
    return np.array(subject_feat, dtype=np.float32)


def load_data(loader, group):
    """加载数据并提取特征"""
    raw_samples = loader.load_all_samples(group=group)
    print(f"加载 {group} 样本: {len(raw_samples)}")
    
    dtw_aligner = DTWAligner(target_length=Config.TARGET_LENGTH)
    feature_extractor = FeatureExtractor()
    
    subject_samples = defaultdict(list)
    subject_fma = {}
    
    for sample in raw_samples:
        if sample.emg_data.size == 0 or sample.kin_data.size == 0:
            continue
        
        try:
            emg_aligned, kin_aligned = dtw_aligner.align(
                sample.emg_data,
                sample.kin_data[:, :63] if sample.kin_data.shape[1] >= 63 else sample.kin_data
            )
            
            emg_feat = feature_extractor.extract_emg_features(emg_aligned)
            kin_feat = feature_extractor.extract_kin_features(kin_aligned)
            feat = np.concatenate([emg_feat, kin_feat])
            
            if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
                continue
            
            sid = f"S{sample.subject_id}"
            subject_samples[sid].append(feat)
            if sample.fma_score is not None:
                subject_fma[sid] = sample.fma_score
                
        except:
            continue
    
    X, y, sids = [], [], []
    for sid, feats in subject_samples.items():
        if len(feats) > 0:
            feats_array = np.array(feats)
            subj_feat = extract_subject_features(feats_array)
            X.append(subj_feat)
            sids.append(sid)
            if sid in subject_fma:
                y.append(subject_fma[sid])
            else:
                y.append(None)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=object), sids


def train_baseline_v1():
    print("="*60)
    print("策略2：参考基准方法 v1.0")
    print("="*60)
    print("\n核心思想：")
    print("  - 健康受试者特征 = 参考基准")
    print("  - 中风患者特征 - 基准 = 相对特征")
    print("  - 相对特征用于FMA预测")
    print("\n目标：MAE<1.40, Acc@1尽可能高")
    print("="*60)
    
    loader = SCIDataLoader(data_root=Config.DATA_DIR)
    
    # 加载健康受试者数据
    print("\n加载健康受试者数据...")
    X_healthy, _, sids_healthy = load_data(loader, "healthy")
    print(f"健康受试者: {len(sids_healthy)} 人, 特征维度: {X_healthy.shape[1]}")
    
    # 加载中风患者数据
    print("\n加载中风患者数据...")
    X_stroke, y_stroke, sids_stroke = load_data(loader, "stroke")
    
    # 过滤有FMA评分的样本
    valid_mask = np.array([y is not None for y in y_stroke])
    X_stroke = X_stroke[valid_mask]
    y_stroke = np.array([y for y in y_stroke if y is not None], dtype=np.float32)
    sids_stroke = [sids_stroke[i] for i in range(len(sids_stroke)) if valid_mask[i]]
    
    print(f"中风受试者: {len(sids_stroke)} 人")
    print(f"FMA分布: {dict(pd.Series(y_stroke).value_counts().sort_index())}")
    
    # 计算健康受试者基准
    print("\n计算健康受试者基准...")
    baseline_mean = np.mean(X_healthy, axis=0)
    baseline_std = np.std(X_healthy, axis=0)
    
    print(f"基准均值范围: [{baseline_mean.min():.3f}, {baseline_mean.max():.3f}]")
    print(f"基准标准差范围: [{baseline_std.min():.3f}, {baseline_std.max():.3f}]")
    
    # 计算相对特征
    print("\n计算相对特征...")
    X_relative = (X_stroke - baseline_mean) / (baseline_std + 1e-10)
    
    print(f"相对特征范围: [{X_relative.min():.3f}, {X_relative.max():.3f}]")
    
    # 分析相对特征与FMA的关系
    print("\n相对特征与FMA的关系:")
    print(f"{'FMA':<8} {'相对特征均值':<15} {'相对特征标准差':<15}")
    print("-" * 40)
    for fma in sorted(set(y_stroke)):
        mask = y_stroke == fma
        print(f"{fma:<8.0f} {X_relative[mask].mean():<15.3f} {X_relative[mask].std():<15.3f}")
    
    # 特征选择
    selector = SelectKBest(f_regression, k=min(50, X_relative.shape[1]-1))
    X_selected = selector.fit_transform(X_relative, y_stroke)
    
    print(f"\n特征选择后维度: {X_selected.shape[1]}")
    
    # PCA降维
    pca = PCA(n_components=min(15, X_selected.shape[1]-1))
    X_pca = pca.fit_transform(X_selected)
    
    print(f"PCA维度: {X_pca.shape[1]}, 解释方差: {np.sum(pca.explained_variance_ratio_):.2%}")
    
    # LOSOCV预测
    print("\n" + "="*60)
    print("LOSOCV预测")
    print("="*60)
    
    loo = LeaveOneOut()
    
    all_preds = []
    all_labels = []
    all_sids = []
    
    for train_idx, test_idx in loo.split(X_pca):
        X_train, X_test = X_pca[train_idx], X_pca[test_idx]
        y_train, y_test = y_stroke[train_idx], y_stroke[test_idx]
        
        # 训练模型
        svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        knn = KNeighborsRegressor(n_neighbors=3)
        gb = GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)
        
        svr.fit(X_train, y_train)
        knn.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        
        # 加权集成
        pred = 0.24 * svr.predict(X_test)[0] + 0.22 * knn.predict(X_test)[0] + 0.54 * gb.predict(X_test)[0]
        
        all_preds.append(pred)
        all_labels.append(y_test[0])
        all_sids.append(sids_stroke[test_idx[0]])
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算指标
    mae = np.mean(np.abs(all_preds - all_labels))
    
    # 四舍五入到FMA值
    pred_round = np.array([Config.FMA_VALUES[np.argmin(np.abs(np.array(Config.FMA_VALUES) - p))] for p in all_preds])
    acc = np.mean(pred_round == all_labels)
    acc_pm1 = np.mean(np.abs(pred_round - all_labels) <= 1)
    
    print(f"\n结果:")
    print(f"  MAE: {mae:.2f}")
    print(f"  Acc@1: {acc:.1%}")
    print(f"  Acc@±1: {acc_pm1:.1%}")
    
    # 按FMA分组分析
    print("\n按FMA分组分析:")
    print(f"{'FMA':<8} {'数量':<8} {'预测正确':<12} {'Acc@1':<10}")
    print("-" * 40)
    for fma in sorted(set(all_labels)):
        mask = all_labels == fma
        n = mask.sum()
        correct = (pred_round[mask] == fma).sum()
        print(f"{fma:<8.0f} {n:<8} {correct:<12} {correct/n:.1%}")
    
    # 保存结果
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    df = pd.DataFrame({
        'subject_id': all_sids,
        'true_fma': all_labels,
        'pred_reg': all_preds,
        'pred_round': pred_round,
    })
    df.to_csv(f'{Config.OUTPUT_DIR}/predictions.csv', index=False)
    
    # 与基线对比
    print("\n" + "="*60)
    print("与基线对比")
    print("="*60)
    print(f"{'方法':<30} {'MAE':<10} {'Acc@1':<10}")
    print("-" * 50)
    print(f"{'v5.7 (基线)':<30} {'1.40':<10} {'25%':<10}")
    print(f"{'v5.8.2 (特征工程)':<30} {'0.96':<10} {'35%':<10}")
    print(f"{'策略1 (异常检测)':<30} {'1.61':<10} {'15%':<10}")
    print(f"{'策略2 (参考基准)':<30} {mae:<10.2f} {acc:.0%}")
    
    # 目标达成判断
    print("\n" + "="*60)
    print("目标达成情况")
    print("="*60)
    if mae < 1.40:
        print(f"[OK] MAE={mae:.2f} < 1.40")
    else:
        print(f"[WARN] MAE={mae:.2f} >= 1.40")
    print(f"Acc@1={acc:.1%}")
    
    return mae, acc, acc_pm1


if __name__ == '__main__':
    mae, acc, acc_pm1 = train_baseline_v1()

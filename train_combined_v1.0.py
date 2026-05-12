# -*- coding: utf-8 -*-
"""
HybridSCIMFP - 组合策略

目标：结合多种策略的最佳实践

组合策略：
1. 使用策略2的Elastic Net特征选择
2. 结合策略3的肌肉协同性特征
3. 优化模型集成权重
4. 尝试多种回归器组合
"""
import os
import sys
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNetCV
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import DataLoader as SCIDataLoader


class Config:
    DATA_DIR = r"C:\Users\Administrator\Desktop\openclaw_cloudcode\sci\dataset\Rocky_Data"
    OUTPUT_DIR = "./outputs/combined_v1.0"
    TARGET_LENGTH = 100
    FMA_VALUES = [14, 16, 17, 18, 19, 20]


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


class SynergyFeatureExtractor:
    """肌肉协同性特征提取器"""
    
    def __init__(self, n_synergies=3):
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


class FeatureExtractor:
    """特征提取器（完整版）"""
    
    def __init__(self, fs_emg=1000, fs_kin=100):
        self.fs_emg = fs_emg
        self.fs_kin = fs_kin
        self.synergy_extractor = SynergyFeatureExtractor(n_synergies=3)
    
    def extract_emg_time_features(self, emg):
        features = []
        for ch in range(emg.shape[1]):
            x = emg[:, ch]
            features.extend([
                np.mean(np.abs(x)),
                np.sqrt(np.mean(x**2)),
                np.var(x),
                np.std(x),
                np.max(x),
                np.min(x),
                np.max(x) - np.min(x),
            ])
            diff = np.diff(x)
            features.extend([
                np.sum(np.abs(diff)) / len(diff),
                np.mean(np.abs(diff)),
                np.std(diff),
            ])
            zero_crossings = np.sum(np.abs(np.diff(np.sign(x)))) / 2
            features.append(zero_crossings / len(x))
            slope_sign_changes = np.sum(np.abs(np.diff(np.sign(diff)))) / 2
            features.append(slope_sign_changes / len(diff) if len(diff) > 0 else 0)
        return np.array(features, dtype=np.float32)
    
    def extract_emg_freq_features(self, emg):
        features = []
        freq_bands = {
            'low': (0, 50),
            'mid_low': (50, 100),
            'mid': (100, 200),
            'mid_high': (200, 350),
            'high': (350, 500)
        }
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
                median_freq = freqs[np.searchsorted(cumsum, cumsum[-1] / 2)]
                features.append(median_freq)
                mean_freq = np.sum(freqs * psd) / (total_power + 1e-10)
                features.append(mean_freq)
                peak_freq = freqs[np.argmax(psd)]
                features.append(peak_freq)
                psd_norm = psd / (total_power + 1e-10)
                spec_entropy = entropy(psd_norm + 1e-10)
                features.append(spec_entropy)
            else:
                features.extend([0] * 9)
        return np.array(features, dtype=np.float32)
    
    def extract_emg_covariance_features(self, emg):
        features = []
        cov_matrix = np.cov(emg.T)
        diag = np.diag(cov_matrix)
        features.extend([
            np.mean(diag),
            np.std(diag),
            np.max(diag),
            np.min(diag),
        ])
        n_channels = emg.shape[1]
        if n_channels > 1:
            off_diag = cov_matrix[np.triu_indices(n_channels, k=1)]
            features.extend([
                np.mean(np.abs(off_diag)),
                np.std(off_diag),
                np.max(off_diag),
                np.min(off_diag),
            ])
            corr_matrix = np.corrcoef(emg.T)
            if not np.isnan(corr_matrix).any():
                off_diag_corr = corr_matrix[np.triu_indices(n_channels, k=1)]
                features.extend([
                    np.mean(off_diag_corr),
                    np.std(off_diag_corr),
                    np.max(off_diag_corr),
                    np.min(off_diag_corr),
                ])
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
    
    def extract_emg_features(self, emg):
        time_feat = self.extract_emg_time_features(emg)
        freq_feat = self.extract_emg_freq_features(emg)
        cov_feat = self.extract_emg_covariance_features(emg)
        nonlinear_feat = self.extract_emg_nonlinear_features(emg)
        synergy_feat = self.synergy_extractor.extract_synergy_features(emg)
        
        return np.concatenate([time_feat, freq_feat, cov_feat, nonlinear_feat, synergy_feat])
    
    def extract_kin_features(self, kin):
        features = []
        features.extend([
            np.mean(kin),
            np.std(kin),
            np.max(kin),
            np.min(kin),
            np.max(kin) - np.min(kin),
        ])
        vel = np.diff(kin, axis=0)
        vel_mag = np.sqrt(np.sum(vel**2, axis=1))
        if len(vel_mag) > 0:
            features.extend([
                np.mean(vel_mag),
                np.std(vel_mag),
                np.max(vel_mag),
            ])
        else:
            features.extend([0, 0, 0])
        if len(vel) > 1:
            acc = np.diff(vel, axis=0)
            acc_mag = np.sqrt(np.sum(acc**2, axis=1))
            if len(acc_mag) > 0:
                features.extend([
                    np.mean(acc_mag),
                    np.std(acc_mag),
                ])
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
                jerk_cost = np.sum(jerk_mag**2) * 0.01
                smoothness = 1 / (1 + jerk_cost)
                features.append(smoothness)
            else:
                features.append(0)
        else:
            features.append(0)
        return np.array(features, dtype=np.float32)


def extract_subject_features(sample_features):
    subject_feat = []
    for i in range(sample_features.shape[1]):
        feat = sample_features[:, i]
        subject_feat.extend([
            np.mean(feat),
            np.std(feat),
            np.min(feat),
            np.max(feat),
            np.median(feat),
            np.percentile(feat, 25),
            np.percentile(feat, 75),
            kurtosis(feat) if len(feat) > 3 else 0,
            skew(feat) if len(feat) > 3 else 0,
        ])
    return np.array(subject_feat, dtype=np.float32)


def load_and_extract_features():
    print("="*60)
    print("加载数据并提取特征（组合策略）...")
    print("="*60)
    
    loader = SCIDataLoader(data_root=Config.DATA_DIR)
    raw_samples = loader.load_all_samples(group="stroke")
    print(f"原始样本: {len(raw_samples)}")
    
    dtw_aligner = DTWAligner(target_length=Config.TARGET_LENGTH)
    feature_extractor = FeatureExtractor(fs_emg=1000, fs_kin=100)
    
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
            
            emg_feat = feature_extractor.extract_emg_features(emg_aligned)
            kin_feat = feature_extractor.extract_kin_features(kin_aligned)
            feat = np.concatenate([emg_feat, kin_feat])
            
            if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
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
            X.append(subj_feat)
            y.append(subject_fma[sid])
            sids.append(sid)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"\n受试者数量: {len(sids)}")
    print(f"受试者级特征维度: {X.shape[1]}")
    print(f"FMA分布: {dict(pd.Series(y).value_counts().sort_index())}")
    
    return X, y, sids


def train_combined_v1():
    print("="*60)
    print("HybridSCIMFP - 组合策略")
    print("="*60)
    print()
    print("组合策略:")
    print("  1. Elastic Net特征选择")
    print("  2. 肌肉协同性特征")
    print("  3. 多模型集成优化")
    print()
    print("目标: MAE<1.40, Acc@1>50%")
    print("="*60)
    
    # 加载数据
    X, y, sids = load_and_extract_features()
    
    # 处理NaN和Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Elastic Net特征选择
    print("\n" + "="*60)
    print("Elastic Net特征选择")
    print("="*60)
    
    elastic = ElasticNetCV(cv=5, random_state=42, max_iter=10000, l1_ratio=0.5)
    elastic.fit(X_scaled, y)
    
    selected_idx = np.where(np.abs(elastic.coef_) > 1e-5)[0]
    print(f"Elastic Net选择特征数: {len(selected_idx)}")
    
    if len(selected_idx) > 0:
        X_selected = X_scaled[:, selected_idx]
    else:
        X_selected = X_scaled
    
    # PCA降维
    pca = PCA(n_components=min(15, X_selected.shape[1] - 1))
    X_pca = pca.fit_transform(X_selected)
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"PCA维度: {X_pca.shape[1]}, 解释方差: {explained_var:.2%}")
    
    # 尝试不同的模型组合
    print("\n" + "="*60)
    print("测试不同模型组合")
    print("="*60)
    
    results = []
    
    # 定义模型组合
    model_combinations = [
        # (名称, 模型列表, 权重列表)
        ("SVR+KNN+GB", 
         [SVR(kernel='rbf', C=1.0, epsilon=0.1),
          KNeighborsRegressor(n_neighbors=3),
          GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)],
         [0.24, 0.22, 0.54]),
        
        ("SVR+KNN+RF",
         [SVR(kernel='rbf', C=1.0, epsilon=0.1),
          KNeighborsRegressor(n_neighbors=3),
          RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)],
         [0.3, 0.3, 0.4]),
        
        ("SVR+Ridge+GB",
         [SVR(kernel='rbf', C=1.0, epsilon=0.1),
          Ridge(alpha=1.0),
          GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)],
         [0.4, 0.2, 0.4]),
        
        ("SVR+Lasso+GB",
         [SVR(kernel='rbf', C=1.0, epsilon=0.1),
          Lasso(alpha=0.1),
          GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)],
         [0.4, 0.2, 0.4]),
        
        ("SVR+KNN+GB+RF",
         [SVR(kernel='rbf', C=1.0, epsilon=0.1),
          KNeighborsRegressor(n_neighbors=3),
          GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42),
          RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)],
         [0.25, 0.25, 0.25, 0.25]),
    ]
    
    for combo_name, models, weights in model_combinations:
        print(f"\n--- {combo_name} ---")
        
        loo = LeaveOneOut()
        all_preds = []
        all_labels = []
        
        for train_idx, test_idx in loo.split(X_pca):
            X_train, X_test_fold = X_pca[train_idx], X_pca[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 训练所有模型
            preds = []
            for model in models:
                model.fit(X_train, y_train)
                preds.append(model.predict(X_test_fold)[0])
            
            # 加权集成
            pred = sum(w * p for w, p in zip(weights, preds))
            
            all_preds.append(pred)
            all_labels.append(y_test[0])
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        mae = np.mean(np.abs(all_preds - all_labels))
        simple_round = np.array([Config.FMA_VALUES[np.argmin(np.abs(np.array(Config.FMA_VALUES) - p))] for p in all_preds])
        acc = np.mean(simple_round == all_labels)
        acc_pm1 = np.mean(np.abs(simple_round - all_labels) <= 1)
        
        print(f"MAE: {mae:.2f}, Acc@1: {acc:.1%}, Acc@±1: {acc_pm1:.1%}")
        
        results.append({
            'name': combo_name,
            'mae': mae,
            'acc': acc,
            'acc_pm1': acc_pm1,
            'preds': all_preds,
            'labels': all_labels,
        })
    
    # 找最佳结果
    print("\n" + "="*60)
    print("结果对比")
    print("="*60)
    
    print(f"\n{'模型组合':<20} {'MAE':<10} {'Acc@1':<10} {'Acc@±1':<10}")
    print("-" * 50)
    
    for r in results:
        print(f"{r['name']:<20} {r['mae']:<10.2f} {r['acc']:.0%}       {r['acc_pm1']:.0%}")
    
    best_result = min(results, key=lambda x: x['mae'])
    
    print(f"\n最佳模型组合: {best_result['name']}")
    print(f"最佳MAE: {best_result['mae']:.2f}")
    print(f"最佳Acc@1: {best_result['acc']:.1%}")
    
    # 与之前结果对比
    print("\n" + "="*60)
    print("与之前结果对比")
    print("="*60)
    
    print(f"\n{'版本':<35} {'MAE':<10} {'Acc@1':<10} {'Acc@±1':<10}")
    print("-" * 65)
    print(f"{'v5.8.2 (特征工程)':<35} {'0.96':<10} {'35%':<10} {'75%':<10}")
    print(f"{'策略2 (Elastic Net)':<35} {'0.72':<10} {'45%':<10} {'85%':<10}")
    print(f"{'组合策略':<35} {best_result['mae']:<10.2f} {best_result['acc']:.0%}       {best_result['acc_pm1']:.0%}")
    
    # 保存结果
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    df = pd.DataFrame({
        'subject_id': sids,
        'true_fma': best_result['labels'],
        'pred_reg': best_result['preds'],
        'pred_round': [Config.FMA_VALUES[np.argmin(np.abs(np.array(Config.FMA_VALUES) - p))] for p in best_result['preds']],
        'error': best_result['preds'] - best_result['labels'],
    })
    df.to_csv(f'{Config.OUTPUT_DIR}/combined_v1.0_predictions.csv', index=False)
    
    summary = pd.DataFrame([{
        'version': 'combined_v1.0',
        'best_model': best_result['name'],
        'mae': best_result['mae'],
        'acc_1': best_result['acc'],
        'acc_pm1': best_result['acc_pm1'],
    }])
    summary.to_csv(f'{Config.OUTPUT_DIR}/combined_v1.0_summary.csv', index=False)
    
    print(f"\n结果已保存: {Config.OUTPUT_DIR}/")
    
    return best_result


if __name__ == '__main__':
    result = train_combined_v1()

# -*- coding: utf-8 -*-
"""
HybridSCIMFP - 策略6：两阶段预测优化

目标：通过分层预测策略提升Acc@1

两阶段策略：
1. 第一阶段：分类器判断FMA区间（低/中/高）
2. 第二阶段：在对应区间内进行回归预测
3. 集成多模型预测
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
from sklearn.decomposition import PCA
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNetCV
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import DataLoader as SCIDataLoader


class Config:
    DATA_DIR = r"C:\Users\Administrator\Desktop\openclaw_cloudcode\sci\dataset\Rocky_Data"
    OUTPUT_DIR = "./outputs/twostage_v1.0"
    TARGET_LENGTH = 100
    FMA_VALUES = [14, 16, 17, 18, 19, 20]
    
    # FMA区间定义
    FMA_GROUPS = {
        'low': [14, 16],      # 低分
        'mid': [17, 18],      # 中分
        'high': [19, 20],     # 高分
    }


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


class FeatureExtractor:
    """特征提取器（v5.8.2版本）"""
    
    def __init__(self, fs_emg=1000, fs_kin=100):
        self.fs_emg = fs_emg
        self.fs_kin = fs_kin
    
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
        return np.concatenate([time_feat, freq_feat, cov_feat, nonlinear_feat])
    
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


def get_fma_group(fma):
    """获取FMA所属区间"""
    for group_name, fma_list in Config.FMA_GROUPS.items():
        if fma in fma_list:
            return group_name
    return 'mid'


def load_and_extract_features():
    print("="*60)
    print("加载数据并提取特征...")
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


def train_twostage_v1():
    print("="*60)
    print("HybridSCIMFP - 策略6：两阶段预测优化")
    print("="*60)
    print()
    print("两阶段策略:")
    print("  1. 第一阶段：分类器判断FMA区间（低/中/高）")
    print("  2. 第二阶段：在对应区间内进行回归预测")
    print("  3. 集成多模型预测")
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
    
    # 使用Elastic Net特征选择
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
    
    # 创建区间标签
    y_groups = np.array([{'low': 0, 'mid': 1, 'high': 2}[get_fma_group(fma)] for fma in y])
    
    print(f"\n区间分布: low={np.sum(y_groups==0)}, mid={np.sum(y_groups==1)}, high={np.sum(y_groups==2)}")
    
    # LOSOCV两阶段预测
    print("\n" + "="*60)
    print("两阶段LOSOCV训练")
    print("="*60)
    
    loo = LeaveOneOut()
    
    all_preds = []
    all_labels = []
    all_group_preds = []
    all_group_labels = []
    
    for train_idx, test_idx in loo.split(X_pca):
        X_train, X_test_fold = X_pca[train_idx], X_pca[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_group_train, y_group_test = y_groups[train_idx], y_groups[test_idx]
        
        # 第一阶段：分类器
        clf_svc = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
        clf_rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        clf_gb = GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=42)
        
        clf_svc.fit(X_train, y_group_train)
        clf_rf.fit(X_train, y_group_train)
        clf_gb.fit(X_train, y_group_train)
        
        # 投票决定区间
        group_probs = (clf_svc.predict_proba(X_test_fold) + 
                       clf_rf.predict_proba(X_test_fold) + 
                       clf_gb.predict_proba(X_test_fold)) / 3
        group_pred = np.argmax(group_probs, axis=1)[0]
        
        all_group_preds.append(group_pred)
        all_group_labels.append(y_group_test[0])
        
        # 第二阶段：回归预测
        svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        knn = KNeighborsRegressor(n_neighbors=3)
        gb = GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)
        
        svr.fit(X_train, y_train)
        knn.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        
        # 加权集成
        w_svr, w_knn, w_gb = 0.24, 0.22, 0.54
        pred_reg = w_svr * svr.predict(X_test_fold)[0] + w_knn * knn.predict(X_test_fold)[0] + w_gb * gb.predict(X_test_fold)[0]
        
        # 根据区间调整预测
        group_fma_ranges = {
            0: (14, 16),  # low
            1: (17, 18),  # mid
            2: (19, 20),  # high
        }
        
        min_fma, max_fma = group_fma_ranges[group_pred]
        
        # 如果预测值在区间内，保持原预测
        # 如果预测值在区间外，调整到区间边界
        if pred_reg < min_fma:
            pred = min_fma + (pred_reg - min_fma) * 0.3  # 软约束
        elif pred_reg > max_fma:
            pred = max_fma + (pred_reg - max_fma) * 0.3
        else:
            pred = pred_reg
        
        all_preds.append(pred)
        all_labels.append(y_test[0])
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_group_preds = np.array(all_group_preds)
    all_group_labels = np.array(all_group_labels)
    
    # 计算指标
    mae = np.mean(np.abs(all_preds - all_labels))
    
    simple_round = np.array([Config.FMA_VALUES[np.argmin(np.abs(np.array(Config.FMA_VALUES) - p))] for p in all_preds])
    acc = np.mean(simple_round == all_labels)
    acc_pm1 = np.mean(np.abs(simple_round - all_labels) <= 1)
    
    # 区间分类准确率
    group_acc = np.mean(all_group_preds == all_group_labels)
    
    print(f"\nMAE: {mae:.2f}")
    print(f"Acc@1: {acc:.1%}")
    print(f"Acc@±1: {acc_pm1:.1%}")
    print(f"区间分类准确率: {group_acc:.1%}")
    
    # 按FMA分组分析
    print("\n" + "="*60)
    print("按FMA分组分析")
    print("="*60)
    
    print(f"\n{'FMA':<8} {'数量':<8} {'预测均值':<12} {'预测标准差':<12} {'平均误差':<12}")
    print("-" * 55)
    
    for fma in sorted(set(all_labels)):
        mask = all_labels == fma
        preds = all_preds[mask]
        errors = preds - fma
        print(f"{fma:<8.0f} {mask.sum():<8} {preds.mean():<12.2f} {preds.std():<12.2f} {errors.mean():<+12.2f}")
    
    # 与之前结果对比
    print("\n" + "="*60)
    print("结果对比")
    print("="*60)
    
    print(f"\n{'版本':<35} {'MAE':<10} {'Acc@1':<10} {'Acc@±1':<10}")
    print("-" * 65)
    print(f"{'v5.8.2 (特征工程)':<35} {'0.96':<10} {'35%':<10} {'75%':<10}")
    print(f"{'策略2 (Elastic Net)':<35} {'0.72':<10} {'45%':<10} {'85%':<10}")
    print(f"{'策略6 (两阶段预测)':<35} {mae:<10.2f} {acc:.0%}       {acc_pm1:.0%}")
    
    # 保存结果
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    df = pd.DataFrame({
        'subject_id': sids,
        'true_fma': all_labels,
        'pred_reg': all_preds,
        'pred_round': [Config.FMA_VALUES[np.argmin(np.abs(np.array(Config.FMA_VALUES) - p))] for p in all_preds],
        'error': all_preds - all_labels,
        'true_group': all_group_labels,
        'pred_group': all_group_preds,
    })
    df.to_csv(f'{Config.OUTPUT_DIR}/twostage_v1.0_predictions.csv', index=False)
    
    summary = pd.DataFrame([{
        'version': 'twostage_v1.0',
        'strategy': '两阶段预测',
        'mae': mae,
        'acc_1': acc,
        'acc_pm1': acc_pm1,
        'group_acc': group_acc,
    }])
    summary.to_csv(f'{Config.OUTPUT_DIR}/twostage_v1.0_summary.csv', index=False)
    
    print(f"\n结果已保存: {Config.OUTPUT_DIR}/")
    
    return {
        'mae': mae,
        'acc': acc,
        'acc_pm1': acc_pm1,
        'group_acc': group_acc,
        'preds': all_preds,
        'labels': all_labels,
    }


if __name__ == '__main__':
    result = train_twostage_v1()

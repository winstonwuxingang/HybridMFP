# -*- coding: utf-8 -*-
"""
策略3-6合并实现 - train_healthy_strategies_v1.0.py

包含：
- 策略3：迁移学习（简化版 - 自编码器特征）
- 策略4：对比学习（简化版 - 距离特征）
- 策略5：恢复程度特征
- 策略6：两阶段分类

目标：MAE<1.40, Acc@1尽可能高
"""
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy.signal import welch
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_regression
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import DataLoader as SCIDataLoader


class Config:
    DATA_DIR = r"C:\Users\Administrator\Desktop\openclaw_cloudcode\sci\dataset\Rocky_Data"
    OUTPUT_DIR = "./outputs/healthy_strategies_v1.0"
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


def evaluate_predictions(all_preds, all_labels, strategy_name):
    """评估预测结果"""
    mae = np.mean(np.abs(all_preds - all_labels))
    pred_round = np.array([Config.FMA_VALUES[np.argmin(np.abs(np.array(Config.FMA_VALUES) - p))] for p in all_preds])
    acc = np.mean(pred_round == all_labels)
    acc_pm1 = np.mean(np.abs(pred_round - all_labels) <= 1)
    
    print(f"\n{strategy_name} 结果:")
    print(f"  MAE: {mae:.2f}")
    print(f"  Acc@1: {acc:.1%}")
    print(f"  Acc@±1: {acc_pm1:.1%}")
    
    return mae, acc, acc_pm1, pred_round


def strategy5_recovery_features(X_healthy, X_stroke, y_stroke, sids_stroke):
    """策略5：恢复程度特征"""
    print("\n" + "="*60)
    print("策略5：恢复程度特征")
    print("="*60)
    
    # 计算健康模板（均值）
    healthy_template = np.mean(X_healthy, axis=0)
    
    # 计算每个中风样本与健康模板的距离
    distances = np.sqrt(np.sum((X_stroke - healthy_template)**2, axis=1))
    
    print(f"恢复程度（距离）范围: [{distances.min():.2f}, {distances.max():.2f}]")
    
    # 将距离作为额外特征
    X_augmented = np.column_stack([X_stroke, distances.reshape(-1, 1)])
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_augmented)
    
    # 特征选择
    selector = SelectKBest(f_regression, k=min(50, X_scaled.shape[1]-1))
    X_selected = selector.fit_transform(X_scaled, y_stroke)
    
    # PCA
    pca = PCA(n_components=min(15, X_selected.shape[1]-1))
    X_pca = pca.fit_transform(X_selected)
    
    # LOSOCV
    loo = LeaveOneOut()
    all_preds, all_labels = [], []
    
    for train_idx, test_idx in loo.split(X_pca):
        X_train, X_test = X_pca[train_idx], X_pca[test_idx]
        y_train, y_test = y_stroke[train_idx], y_stroke[test_idx]
        
        svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        gb = GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)
        
        svr.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        
        pred = 0.5 * svr.predict(X_test)[0] + 0.5 * gb.predict(X_test)[0]
        all_preds.append(pred)
        all_labels.append(y_test[0])
    
    return evaluate_predictions(np.array(all_preds), np.array(all_labels), "策略5")


def strategy6_two_stage(X_healthy, X_stroke, y_stroke, sids_stroke):
    """策略6：两阶段分类"""
    print("\n" + "="*60)
    print("策略6：两阶段分类")
    print("="*60)
    
    # 标准化
    scaler = StandardScaler()
    X_healthy_scaled = scaler.fit_transform(X_healthy)
    X_stroke_scaled = scaler.transform(X_stroke)
    
    # 阶段1：训练健康/中风分类器
    X_all = np.concatenate([X_healthy_scaled, X_stroke_scaled])
    y_binary = np.concatenate([np.zeros(len(X_healthy)), np.ones(len(X_stroke))])
    
    binary_clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    binary_clf.fit(X_all, y_binary)
    
    print(f"健康/中风分类器训练完成")
    
    # 阶段2：训练FMA预测器
    selector = SelectKBest(f_regression, k=min(50, X_stroke_scaled.shape[1]-1))
    X_selected = selector.fit_transform(X_stroke_scaled, y_stroke)
    
    pca = PCA(n_components=min(15, X_selected.shape[1]-1))
    X_pca = pca.fit_transform(X_selected)
    
    # LOSOCV
    loo = LeaveOneOut()
    all_preds, all_labels = [], []
    
    for train_idx, test_idx in loo.split(X_pca):
        X_train, X_test = X_pca[train_idx], X_pca[test_idx]
        y_train, y_test = y_stroke[train_idx], y_stroke[test_idx]
        
        # 分类器
        clf = GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=42)
        # 回归器
        reg = GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)
        
        # 将FMA转换为类别索引
        y_train_class = np.array([Config.FMA_VALUES.index(int(f)) for f in y_train])
        
        clf.fit(X_train, y_train_class)
        reg.fit(X_train, y_train)
        
        # 分类预测
        pred_class = Config.FMA_VALUES[clf.predict(X_test)[0]]
        # 回归预测
        pred_reg = reg.predict(X_test)[0]
        
        # 加权融合
        pred = 0.6 * pred_class + 0.4 * pred_reg
        
        all_preds.append(pred)
        all_labels.append(y_test[0])
    
    return evaluate_predictions(np.array(all_preds), np.array(all_labels), "策略6")


def strategy_combined(X_healthy, X_stroke, y_stroke, sids_stroke):
    """组合策略：策略5 + 策略6"""
    print("\n" + "="*60)
    print("组合策略：策略5 + 策略6")
    print("="*60)
    
    # 策略5：恢复程度特征
    healthy_template = np.mean(X_healthy, axis=0)
    distances = np.sqrt(np.sum((X_stroke - healthy_template)**2, axis=1))
    X_augmented = np.column_stack([X_stroke, distances.reshape(-1, 1)])
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_augmented)
    
    # 特征选择 + PCA
    selector = SelectKBest(f_regression, k=min(50, X_scaled.shape[1]-1))
    X_selected = selector.fit_transform(X_scaled, y_stroke)
    
    pca = PCA(n_components=min(15, X_selected.shape[1]-1))
    X_pca = pca.fit_transform(X_selected)
    
    # LOSOCV + 两阶段预测
    loo = LeaveOneOut()
    all_preds, all_labels = [], []
    
    for train_idx, test_idx in loo.split(X_pca):
        X_train, X_test = X_pca[train_idx], X_pca[test_idx]
        y_train, y_test = y_stroke[train_idx], y_stroke[test_idx]
        
        # 分类器
        clf = GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=42)
        # 回归器
        reg = GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)
        
        y_train_class = np.array([Config.FMA_VALUES.index(int(f)) for f in y_train])
        
        clf.fit(X_train, y_train_class)
        reg.fit(X_train, y_train)
        
        pred_class = Config.FMA_VALUES[clf.predict(X_test)[0]]
        pred_reg = reg.predict(X_test)[0]
        
        # 加权融合
        pred = 0.6 * pred_class + 0.4 * pred_reg
        
        all_preds.append(pred)
        all_labels.append(y_test[0])
    
    return evaluate_predictions(np.array(all_preds), np.array(all_labels), "组合策略")


def main():
    print("="*60)
    print("策略3-6合并实现 v1.0")
    print("="*60)
    print("\n目标：MAE<1.40, Acc@1尽可能高")
    print("="*60)
    
    loader = SCIDataLoader(data_root=Config.DATA_DIR)
    
    # 加载数据
    print("\n加载数据...")
    X_healthy, _, sids_healthy = load_data(loader, "healthy")
    X_stroke, y_stroke, sids_stroke = load_data(loader, "stroke")
    
    # 过滤有FMA评分的样本
    valid_mask = np.array([y is not None for y in y_stroke])
    X_stroke = X_stroke[valid_mask]
    y_stroke = np.array([y for y in y_stroke if y is not None], dtype=np.float32)
    sids_stroke = [sids_stroke[i] for i in range(len(sids_stroke)) if valid_mask[i]]
    
    print(f"\n健康受试者: {len(sids_healthy)} 人")
    print(f"中风受试者: {len(sids_stroke)} 人")
    print(f"FMA分布: {dict(pd.Series(y_stroke).value_counts().sort_index())}")
    
    # 执行各策略
    results = {}
    
    # 策略5
    mae5, acc5, acc_pm1_5, _ = strategy5_recovery_features(X_healthy, X_stroke, y_stroke, sids_stroke)
    results['策略5'] = {'mae': mae5, 'acc': acc5, 'acc_pm1': acc_pm1_5}
    
    # 策略6
    mae6, acc6, acc_pm1_6, _ = strategy6_two_stage(X_healthy, X_stroke, y_stroke, sids_stroke)
    results['策略6'] = {'mae': mae6, 'acc': acc6, 'acc_pm1': acc_pm1_6}
    
    # 组合策略
    mae_comb, acc_comb, acc_pm1_comb, _ = strategy_combined(X_healthy, X_stroke, y_stroke, sids_stroke)
    results['组合策略'] = {'mae': mae_comb, 'acc': acc_comb, 'acc_pm1': acc_pm1_comb}
    
    # 最终对比
    print("\n" + "="*60)
    print("所有策略对比")
    print("="*60)
    print(f"\n{'方法':<25} {'MAE':<10} {'Acc@1':<10} {'Acc@±1':<10}")
    print("-" * 55)
    print(f"{'v5.7 (基线)':<25} {'1.40':<10} {'25%':<10} {'-':<10}")
    print(f"{'v5.8.2 (特征工程)':<25} {'0.96':<10} {'35%':<10} {'75%':<10}")
    print(f"{'策略1 (异常检测)':<25} {'1.61':<10} {'15%':<10} {'55%':<10}")
    print(f"{'策略2 (参考基准)':<25} {'1.72':<10} {'25%':<10} {'45%':<10}")
    print(f"{'策略5 (恢复程度)':<25} {mae5:<10.2f} {acc5:.0%}       {acc_pm1_5:.0%}")
    print(f"{'策略6 (两阶段分类)':<25} {mae6:<10.2f} {acc6:.0%}       {acc_pm1_6:.0%}")
    print(f"{'组合策略':<25} {mae_comb:<10.2f} {acc_comb:.0%}       {acc_pm1_comb:.0%}")
    
    # 保存结果
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    df = pd.DataFrame([
        {'策略': '策略5', 'MAE': mae5, 'Acc@1': acc5, 'Acc@±1': acc_pm1_5},
        {'策略': '策略6', 'MAE': mae6, 'Acc@1': acc6, 'Acc@±1': acc_pm1_6},
        {'策略': '组合策略', 'MAE': mae_comb, 'Acc@1': acc_comb, 'Acc@±1': acc_pm1_comb},
    ])
    df.to_csv(f'{Config.OUTPUT_DIR}/all_results.csv', index=False)
    
    return results


if __name__ == '__main__':
    results = main()

# -*- coding: utf-8 -*-
"""
v5.8 - 基于v5.7特征的准确率优化
目标：MAE≈1.40, Acc@1≥70%

策略：
1. 使用v5.7的最佳特征和模型
2. 优化决策阈值
3. 分析理论上限
"""
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import DataLoader as SCIDataLoader


class Config:
    DATA_DIR = r"C:\Users\Administrator\Desktop\openclaw_cloudcode\sci\dataset\Rocky_Data"
    OUTPUT_DIR = "./outputs/v5.8"
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


class PriviteraFeatureExtractor:
    def __init__(self, fs_emg=1000, fs_kin=100):
        self.fs_emg = fs_emg
        self.fs_kin = fs_kin
    
    def extract_emg_features(self, emg):
        features = []
        for ch in range(emg.shape[1]):
            x = emg[:, ch]
            features.extend([
                np.mean(np.abs(x)),
                np.sqrt(np.mean(x**2)),
                np.var(x),
                np.sum(np.abs(np.diff(x))) / len(x),
                np.max(x),
                np.min(x),
                np.std(x),
            ])
        for ch in range(emg.shape[1]):
            x = emg[:, ch]
            freqs = np.fft.rfftfreq(len(x), 1/self.fs_emg)
            psd = np.abs(np.fft.rfft(x))**2
            if len(psd) > 0 and np.sum(psd) > 0:
                features.append(np.sum(freqs * psd) / (np.sum(psd) + 1e-10))
            else:
                features.append(0.0)
        features.append(np.mean(np.abs(emg)))
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
        if len(vel) > 1:
            acc = np.diff(vel, axis=0)
            acc_mag = np.sqrt(np.sum(acc**2, axis=1))
            features.extend([
                np.mean(acc_mag) if len(acc_mag) > 0 else 0,
                np.std(acc_mag) if len(acc_mag) > 0 else 0,
                np.max(acc_mag) if len(acc_mag) > 0 else 0,
            ])
        else:
            features.extend([0, 0, 0])
        if len(kin) > 3:
            vel = np.gradient(kin, axis=0)
            acc = np.gradient(vel, axis=0)
            jerk = np.gradient(acc, axis=0)
            jerk_mag = np.sqrt(np.sum(jerk**2, axis=1))
            features.append(1 / (1 + np.sum(jerk_mag) / len(jerk_mag)))
        else:
            features.append(0)
        diff = np.diff(kin, axis=0)
        features.append(np.sum(np.sqrt(np.sum(diff**2, axis=1))))
        features.extend([np.mean(kin), np.std(kin), np.max(kin), np.min(kin)])
        features.append(np.max(kin) - np.min(kin))
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
        ])
    return np.array(subject_feat, dtype=np.float32)


def load_data():
    print("加载数据...")
    loader = SCIDataLoader(data_root=Config.DATA_DIR)
    raw_samples = loader.load_all_samples(group="stroke")
    print(f"原始样本: {len(raw_samples)}")
    
    dtw_aligner = DTWAligner(target_length=Config.TARGET_LENGTH)
    feature_extractor = PriviteraFeatureExtractor(fs_emg=1000, fs_kin=100)
    
    subject_samples = defaultdict(list)
    subject_fma = {}
    
    print("特征提取中...")
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
            
        except:
            continue
    
    # 提取受试者级别特征
    X, y, sids = [], [], []
    for sid, feats in subject_samples.items():
        if len(feats) > 0:
            feats_array = np.array(feats)
            subj_feat = extract_subject_features(feats_array)
            X.append(subj_feat)
            y.append(subject_fma[sid])
            sids.append(sid)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), sids


def theoretical_analysis(y):
    """理论上限分析"""
    print("\n" + "="*60)
    print("理论上限分析")
    print("="*60)
    
    n = len(y)
    total_error_budget = 1.40 * n  # MAE=1.40时的总误差预算
    
    print(f"\n总样本数: {n}")
    print(f"MAE=1.40时的总误差预算: {total_error_budget:.0f}")
    
    # 计算不同Acc@1下的理论MAE
    print(f"\n{'Acc@1':<10} {'精确匹配数':<12} {'剩余样本':<10} {'剩余样本平均误差':<20} {'理论MAE':<10} {'可行性':<10}")
    print("-" * 80)
    
    for acc in [0.50, 0.60, 0.70, 0.75, 0.80]:
        n_correct = int(acc * n)
        n_remaining = n - n_correct
        avg_error_remaining = total_error_budget / n_remaining if n_remaining > 0 else 0
        
        # 检查可行性（最大误差=6）
        feasible = "可行" if avg_error_remaining <= 6 else "不可行"
        
        print(f"{acc:.0%}       {n_correct:<12} {n_remaining:<10} {avg_error_remaining:<20.2f} {total_error_budget/n:.2f}       {feasible}")
    
    # 更精确的分析
    print("\n精确分析（假设误差分布）:")
    print(f"{'Acc@1':<10} {'精确匹配':<10} {'误差=1':<10} {'误差=2':<10} {'误差=3':<10} {'误差=4+':<10} {'MAE':<10}")
    print("-" * 70)
    
    for acc in [0.50, 0.60, 0.70]:
        n_correct = int(acc * n)
        n_remaining = n - n_correct
        
        # 假设剩余样本的误差分布
        for err_dist in [(1, 0.5, 0.3, 0.2), (1, 0.6, 0.3, 0.1), (1, 0.4, 0.4, 0.2)]:
            # err_dist: (误差=1的比例, 误差=2的比例, 误差=3的比例, 误差=4+的比例)
            n_err1 = int(n_remaining * err_dist[0])
            n_err2 = int(n_remaining * err_dist[1])
            n_err3 = int(n_remaining * err_dist[2])
            n_err4 = n_remaining - n_err1 - n_err2 - n_err3
            
            mae = (n_correct * 0 + n_err1 * 1 + n_err2 * 2 + n_err3 * 3 + n_err4 * 4) / n
            
            if abs(mae - 1.40) < 0.1:
                print(f"{acc:.0%}       {n_correct:<10} {n_err1:<10} {n_err2:<10} {n_err3:<10} {n_err4:<10} {mae:<10.2f} <- 接近目标")


def train_v58():
    print("="*60)
    print("v5.8 - 基于v5.7特征的准确率优化")
    print("目标: MAE=1.40, Acc@1>=70%")
    print("="*60)
    
    # 加载数据
    X, y, sids = load_data()
    
    print(f"\n受试者: {len(sids)}")
    print(f"特征维度: {X.shape[1]}")
    print(f"FMA分布: {dict(pd.Series(y).value_counts().sort_index())}")
    
    # 理论分析
    theoretical_analysis(y)
    
    # 标准化 + PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=15)
    X_pca = pca.fit_transform(X_scaled)
    print(f"\nPCA解释方差: {np.sum(pca.explained_variance_ratio_):.2%}")
    
    # LOSOCV
    loo = LeaveOneOut()
    
    all_preds = []
    all_labels = []
    all_sids = []
    
    print("\nLOSOCV预测中...")
    for train_idx, test_idx in loo.split(X_pca):
        X_train, X_test = X_pca[train_idx], X_pca[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Top-3模型（与v5.7相同）
        svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        knn = KNeighborsRegressor(n_neighbors=3)
        gb = GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)
        
        svr.fit(X_train, y_train)
        knn.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        
        # 加权集成（与v5.7相同）
        w_svr, w_knn, w_gb = 0.24, 0.22, 0.54
        pred = w_svr * svr.predict(X_test)[0] + w_knn * knn.predict(X_test)[0] + w_gb * gb.predict(X_test)[0]
        
        all_preds.append(pred)
        all_labels.append(y_test[0])
        all_sids.append(sids[test_idx[0]])
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 基线结果
    mae_baseline = np.mean(np.abs(all_preds - all_labels))
    
    # 简单四舍五入
    simple_round = np.array([Config.FMA_VALUES[np.argmin(np.abs(np.array(Config.FMA_VALUES) - p))] for p in all_preds])
    acc_simple = np.mean(simple_round == all_labels)
    mae_simple = np.mean(np.abs(simple_round - all_labels))
    
    print(f"\n基线结果（与v5.7相同）:")
    print(f"  回归MAE: {mae_baseline:.2f}")
    print(f"  四舍五入后MAE: {mae_simple:.2f}")
    print(f"  Acc@1: {acc_simple:.1%}")
    
    # 详细预测分析
    print("\n" + "="*60)
    print("详细预测分析")
    print("="*60)
    
    print(f"\n{'Subject':<10} {'True':<8} {'Pred':<10} {'Round':<8} {'Error':<10} {'Status':<10}")
    print("-" * 60)
    
    for i in range(len(all_sids)):
        err = all_preds[i] - all_labels[i]
        status = "OK" if simple_round[i] == all_labels[i] else "WRONG"
        print(f"{all_sids[i]:<10} {all_labels[i]:<8.0f} {all_preds[i]:<10.2f} {simple_round[i]:<8.0f} {err:<+10.2f} {status:<10}")
    
    # 误差分布分析
    print("\n" + "="*60)
    print("误差分布分析")
    print("="*60)
    
    errors = all_preds - all_labels
    abs_errors = np.abs(errors)
    
    print(f"\n按FMA分组:")
    print(f"{'FMA':<8} {'数量':<8} {'平均误差':<12} {'预测偏高':<12} {'预测偏低':<12}")
    print("-" * 55)
    
    for fma in sorted(set(all_labels)):
        mask = all_labels == fma
        n = mask.sum()
        mean_err = errors[mask].mean()
        n_high = (errors[mask] > 0.5).sum()
        n_low = (errors[mask] < -0.5).sum()
        print(f"{fma:<8.0f} {n:<8} {mean_err:<+12.2f} {n_high:<12} {n_low:<12}")
    
    # Acc@1的理论上限
    print("\n" + "="*60)
    print("Acc@1提升策略分析")
    print("="*60)
    
    # 计算每个预测到真实FMA的距离
    distances = np.array([min([abs(p - f) for f in Config.FMA_VALUES]) for p in all_preds])
    
    print(f"\n预测值到最近FMA的距离分布:")
    print(f"  距离<0.5: {(distances < 0.5).sum()} 个（可精确匹配）")
    print(f"  距离0.5-1.0: {((distances >= 0.5) & (distances < 1.0)).sum()} 个（可能匹配）")
    print(f"  距离1.0-2.0: {((distances >= 1.0) & (distances < 2.0)).sum()} 个（难以匹配）")
    print(f"  距离>2.0: {(distances >= 2.0).sum()} 个（无法匹配）")
    
    # 理论最大Acc@1
    max_acc = (distances < 0.5).sum() / len(distances)
    print(f"\n理论最大Acc@1（距离<0.5的样本）: {max_acc:.1%}")
    
    # 如果允许调整阈值
    print("\n阈值优化策略:")
    
    # 找出每个FMA类别的预测范围
    print(f"\n{'FMA':<8} {'预测范围':<20} {'预测均值':<12} {'预测标准差':<12}")
    print("-" * 55)
    
    for fma in sorted(set(all_labels)):
        mask = all_labels == fma
        preds = all_preds[mask]
        print(f"{fma:<8.0f} [{preds.min():.1f}, {preds.max():.1f}]{'':<5} {preds.mean():<12.2f} {preds.std():<12.2f}")
    
    # 结论
    print("\n" + "="*60)
    print("结论")
    print("="*60)
    
    print(f"""
基于当前预测结果分析：

1. 当前状态：
   - 回归MAE: {mae_baseline:.2f}
   - Acc@1: {acc_simple:.1%}
   - 理论最大Acc@1: {max_acc:.1%}

2. 问题诊断：
   - 低FMA样本（14,16）预测偏高
   - 高FMA样本（20）预测偏低
   - 预测值集中在17-19区间

3. Acc@1=70%的可行性：
   - 需要14个样本精确匹配
   - 当前仅{(distances < 0.5).sum()}个样本预测距离<0.5
   - 需要改善{14 - (distances < 0.5).sum()}个样本的预测

4. 建议：
   - 在MAE=1.40约束下，Acc@1=70%理论可行但实际困难
   - 根本问题：低FMA样本特征与高FMA相似
   - 需要更多数据或更好的特征工程
""")
    
    # 保存结果
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    df = pd.DataFrame({
        'subject_id': all_sids,
        'true_fma': all_labels,
        'pred_reg': all_preds,
        'pred_round': simple_round,
        'error': errors,
        'abs_error': abs_errors,
        'distance_to_fma': distances
    })
    df.to_csv(f'{Config.OUTPUT_DIR}/v5.8_detailed_analysis.csv', index=False)
    
    print(f"\n详细结果已保存: {Config.OUTPUT_DIR}/v5.8_detailed_analysis.csv")
    
    return mae_baseline, acc_simple, max_acc


if __name__ == '__main__':
    mae, acc, max_acc = train_v58()

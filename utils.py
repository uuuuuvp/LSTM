import torch
import numpy as np
import random
import tqdm
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def get_rolling_window_multistep(forecasting_length, interval_length, window_length, features, labels):
    output_features = np.zeros((1, features.shape[0], window_length))
    output_labels = np.zeros((1, 1, forecasting_length))
    if features.shape[1] != labels.shape[1]:
        assert 'cant process such data'
    else:
        output_features = np.zeros((1, features.shape[0], window_length))
        output_labels = np.zeros((1, 1, forecasting_length))
        for index in tqdm.tqdm(range(0, features.shape[1]-interval_length-window_length-forecasting_length+1), desc='data preparing'):
            output_features = np.concatenate((output_features, np.expand_dims(features[:, index:index+window_length], axis=0)))
            output_labels = np.concatenate((output_labels, np.expand_dims(labels[:, index+interval_length+window_length: index+interval_length+window_length+forecasting_length], axis=0)))
    output_features = output_features[1:, :, :]
    output_labels = output_labels[1:, :, :]
    return torch.from_numpy(output_features), torch.from_numpy(output_labels)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def Otl_Plt(df, target_col, method='mad', threshold=5.0, lag=84, limit=3, outlier_flag=True, interpolation=False):
    """
    专门针对电力功率数据的清洗函数
    :param df: 数据框
    :param target_col: 目标功率列名
    :param method: 异常检测方法 'mad' (推荐) 或 'sigma'
    :param threshold: 阈值，mad默认为5，sigma默认为3
    :param lag: 周期性填充的偏移量（如15min采样，一天为96点）
    :param limit: 连续空洞插值的最大长度
    :return: 清洗后的df, 异常值比例
    """
    data = df[target_col].copy()
    initial_nan_count = data.isna().sum()

    # ---- 1. 异常值检测 ----
    if method == 'mad':
        median = data.median()
        mad = np.median(np.abs(data - median))
        # 避免mad为0导致除法问题
        if mad == 0:
            mad = data.std()
        lower_limit = median - threshold * mad
        upper_limit = median + threshold * mad
    elif method == 'sigma':  # sigma 方法
        mean = data.mean()
        std = data.std()
        lower_limit = mean - threshold * std
        upper_limit = mean + threshold * std
    elif method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        # threshold 作为 IQR 的乘数，默认 1.5 是轻度，3.0 是重度
        lower_limit = Q1 - threshold * IQR
        upper_limit = Q3 + threshold * IQR

    if outlier_flag:
        # 将异常值标记为 NaN
        data.loc[(data > upper_limit) | (data < lower_limit)] = np.nan
    
    # 计算新增的 NaN 比例（即异常值占比）
    outlier_ratio = (data.isna().sum() - initial_nan_count) / len(data)

    # ---- 2. 分层填充逻辑 ----
    if interpolation:
        # 第一层：短洞线性插值（处理孤立点）
        data = data.interpolate(method='linear', limit=limit)

        # 第二层：长洞周期性填充（利用 lag 找昨天或上个周期的对应点）
        # 如果数据量足够大，尝试用 shift(lag) 填充剩下的 NaN
        if len(data) > lag:
            data = data.fillna(data.shift(lag))
    
        # 第三层：兜底填充（前向+后向）
        data = data.ffill().bfill()

    df[target_col] = data
    return df, outlier_ratio

def safe_mape(y_true, y_pred, epsilon=0.01):
        y_true_safe = np.where(y_true == 0, epsilon, y_true)
        return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
def Mmape(y_true, y_pred, threshold_ratio=0.05):
    # 针对 MAPE 的特殊处理：Masked MAPE
    # 计算当前测试集的最大值，确定过滤门限
    max_val = np.max(y_true)
    mask = y_true > (max_val * threshold_ratio)
    if np.any(mask):
        # 只计算真值大于门限值的点的百分比误差
        # 这能真实反映线路“带载”状态下的预测精度
        mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask])
    else:
        # 如果整段数据几乎都是 0（极度空载），给分母加一个极小偏置防止爆炸
        mape = mean_absolute_percentage_error(y_true + 1e-6, y_pred + 1e-6)

    return mape
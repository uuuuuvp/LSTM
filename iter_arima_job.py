import os
import pandas as pd
import numpy as np
import warnings
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import Mmape, read_data_from_csv 

warnings.filterwarnings("ignore")

# ================== 实验模式配置 ==================
# 想要跑“小时级”就写 'hour'，跑“天级别”就写 'day'
MODE = 'hour' 

if MODE == 'hour':
    TRAIN_POINTS = 1488           # 训练点数
    FORECAST_HORIZON = 168        # 预测总长度 (14天)
    SCALES = [12, 24, 36, 84, 168] # 评价尺度
    DAY_FLAG = False              # 不降采样
    M_VAL = 12                    # 日周期为 12 (2h一拍)
else:
    TRAIN_POINTS = 110            # 训练天数
    FORECAST_HORIZON = 28         # 预测天数 (4周)
    SCALES = [7, 14, 21, 28]      # 评价尺度 (单位:天)
    DAY_FLAG = True               # 开启天级别降采样
    M_VAL = 7                     # 周周期为 7 (每天一拍)

# ================== 通用路径设置 ==================
DATA_DIRECTORY = "E:/data/output_lines_0-6/"
# INPUT_FILE = r"E:\Downloads\基于LSTM时间序列预测\基于LSTM时间序列预测\LSTM\Exp\LineIndex\full_sampled_lines.csv"
INPUT_FILE = r"E:\Downloads\基于LSTM时间序列预测\基于LSTM时间序列预测\LSTM\Exp\LineIndex\aic_storage.csv"
COLUMN_NAME = 'I_P'
CRITERION = 'aic' 

# 动态生成文件名，防止数据覆盖
OUTPUT_FILE = f"E:\\Downloads\\基于LSTM时间序列预测\\基于LSTM时间序列预测\\LSTM\\Exp\\ARIMAResult\\TheLast_arima-{MODE}-{TRAIN_POINTS}-{CRITERION}-{COLUMN_NAME}.csv"
CHECKPOINT_FILE = f"checkpoint_arima_{MODE}.csv"
SAVE_INTERVAL = 10 
WORKERS = 6

# ================== 逻辑函数 ==================

def save_results(results_list, output_file, scales, is_checkpoint=False):
    if not results_list: return
    df = pd.DataFrame(results_list)
    base_cols = ['line_name', 'p', 'd', 'q', 'status']
    metric_cols = []
    for s in scales:
        metric_cols.extend([f'mae_{s}', f'mse_{s}', f'mape_{s}', f'r2_{s}'])
    
    all_expected_cols = base_cols + metric_cols
    for col in all_expected_cols:
        if col not in df.columns: df[col] = None
            
    df[all_expected_cols].to_csv(output_file, index=False, encoding='utf-8-sig')
    if is_checkpoint: print(f"   💾 检查点保存: {len(results_list)} 条")

def process_single_line(line_info):
    """
    解包顺序必须与下方 process_args 一致
    """
    line_name, train_pts, f_horizon, crit, data_dir, col_name, day_flag, m_val, scales = line_info
    
    try:
        file_path = os.path.join(data_dir, f"{line_name}.csv")
        if not os.path.exists(file_path): return None
        
        # 1. 获取数据
        train, test = read_data_from_csv(
            file_name=file_path, column_name=col_name,
            train_points=train_pts, forecast_horizon=f_horizon,
            resample_to_day=day_flag
        )
        
        if len(test) < max(scales):
            return {'line_name': line_name, 'status': f'failed: len_{len(test)}_too_short'}

        # 2. 变换
        pt = PowerTransformer(method='yeo-johnson')
        train_transformed = pt.fit_transform(train.reshape(-1, 1)).flatten()
        
        # 3. 拟合
        model = auto_arima(
            train_transformed, seasonal=True, m=m_val,
            information_criterion=crit, suppress_warnings=True, error_action='ignore'
        )
        p, d, q = model.order
        
        # 4. 滚动预测
        y_pred_list = []
        for t in range(len(test)):
            y_hat_trans = model.predict(n_periods=1)[0]
            y_hat = pt.inverse_transform(np.array([[y_hat_trans]]))[0, 0]
            y_pred_list.append(y_hat)
            
            # 更新真值
            actual_trans = pt.transform(np.array([[test[t]]]))[0, 0]
            model.update([actual_trans])
            
        y_pred, y_true = np.array(y_pred_list), test
        
        # 5. 计算指标
        line_res = {'line_name': line_name, 'p': p, 'd': d, 'q': q, 'status': 'success'}
        for s in scales:
            s_true, s_pred = y_true[:s], y_pred[:s]
            line_res[f'mae_{s}'] = mean_absolute_error(s_true, s_pred)
            line_res[f'mse_{s}'] = mean_squared_error(s_true, s_pred)
            line_res[f'r2_{s}'] = r2_score(s_true, s_pred)
            line_res[f'mape_{s}'] = Mmape(s_true, s_pred)
            
        return line_res
    except Exception as e:
        return {'line_name': line_name, 'status': f'failed: {str(e)}'}

if __name__ == "__main__":
    lines_df = pd.read_csv(INPUT_FILE)
    all_line_names = lines_df['line_name'].tolist()
    
    # 这里加一个 load_checkpoint 逻辑（略，见之前代码）
    results_list = [] 
    pending_lines = all_line_names # 简写，实际可用 checkpoint 过滤
    
    # 【传参核心】顺序必须严格对应
    process_args = [
        (name, TRAIN_POINTS, FORECAST_HORIZON, CRITERION, DATA_DIRECTORY, COLUMN_NAME, DAY_FLAG, M_VAL, SCALES)
        for name in pending_lines
    ]

    print(f"🚀 启动模式: {MODE.upper()} | 周期: {M_VAL} | 进程: {WORKERS}")
    
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        future_to_line = {executor.submit(process_single_line, args): args[0] for args in process_args}
        
        for idx, future in enumerate(as_completed(future_to_line), 1):
            line_name = future_to_line[future]
            try:
                res = future.result()
                if res: results_list.append(res)
                if len(results_list) % SAVE_INTERVAL == 0:
                    save_results(results_list, CHECKPOINT_FILE, SCALES, is_checkpoint=True)
                print(f"[{idx}/{len(all_line_names)}] {line_name} - OK")
            except Exception as e:
                print(f"Error {line_name}: {e}")

    save_results(results_list, OUTPUT_FILE, SCALES)
    print(f"✅ 完成！结果路径: {OUTPUT_FILE}")
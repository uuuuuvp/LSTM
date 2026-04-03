import os
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

# 从你的 utils 中引用 Mmape 和 read_data_from_csv
# 假设你的文件名是 utils.py，如果不是请修改下行
from utils import Mmape, read_data_from_csv 

warnings.filterwarnings("ignore")

# ================== 参数设置 ==================
data_directory = "E:/data/output_lines_0-6/"
input_file = "E:\\code\\mission\\full_sampled_lines.csv"
column_name = 'I_P'  # 保持与 LSTM 实验一致
train_points = 800
forecast_horizon = 12
sigma_threshold = 3

# ARIMA 参数范围（bic 通常比 aic 更倾向于简单模型，防止过拟合）
criterion = 'bic' 
output_file = f"arima_vs_lstm_bench_{column_name}.csv"

# ================== 读取数据与初始化 ==================
lines_df = pd.read_csv(input_file)
results_df = lines_df.copy()

# 统一指标列名，方便对比
for col in ['mae', 'mse', 'mape', 'r2', 'p', 'd', 'q', 'status']:
    results_df[col] = None

# ================== 循环处理每条线路 ==================
for idx, row in results_df.iterrows():
    line_name = row['line_name']
    print(f"\n[{idx+1}/{len(results_df)}] ARIMA 对标测试: {line_name}")
    
    file_path = os.path.join(data_directory, line_name + '.csv')
    
    if not os.path.exists(file_path):
        results_df.loc[idx, 'status'] = 'file_not_found'
        continue
    
    try:
        # 1. 使用你提供的函数读取数据
        train, test = read_data_from_csv(
            file_name=file_path,
            column_name=column_name,
            train_points=train_points,
            forecast_horizon=forecast_horizon,
            sigma_threshold=sigma_threshold
        )
        
        # 2. 自动寻找最优 ARIMA 模型
        # m=12 代表季节性周期（一天12个点，每2小时一个点）
        model = auto_arima(
            train,
            seasonal=True, m=12,
            information_criterion=criterion,
            suppress_warnings=True,
            error_action='ignore',
            stepwise=True
        )
        
        # 3. 预测未来 12 个点
        y_pred = model.predict(n_periods=forecast_horizon)
        y_true = test
        
        # 4. 调用统一指标
        mse_l = mean_squared_error(y_true, y_pred)
        mae_l = mean_absolute_error(y_true, y_pred)
        r2_l = r2_score(y_true, y_pred)
        # 调用你 utils 里的 Mmape
        mape_l = Mmape(y_true, y_pred, threshold_ratio=0.05)
        
        # 5. 记录参数和结果
        p, d, q = model.order
        results_df.loc[idx, ['mae', 'mse', 'mape', 'r2', 'p', 'd', 'q', 'status']] = \
            [mae_l, mse_l, mape_l, r2_l, p, d, q, 'success']
        
        print(f"   结果: R2={r2_l:.4f} | Mmape={mape_l:.4f} | Order({p},{d},{q})")
        
    except Exception as e:
        print(f"   出错: {str(e)}")
        results_df.loc[idx, 'status'] = f'failed: {str(e)}'
    
    # 进度保存
    if (idx + 1) % 5 == 0:
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print("\nARIMA 对标实验结束。")
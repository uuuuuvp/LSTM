import os
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer
import warnings
from utils import Mmape, read_data_from_csv 

warnings.filterwarnings("ignore")

# ================== 参数设置 ==================
data_directory = "E:/data/output_lines_0-6/"
input_file = "E:\\code\\mission\\full_sampled_lines.csv"
column_name = 'I_P'
train_points = 1600
forecast_horizon = 160  # 依然在测试集的 12 个点上比，但是是一个一个测
criterion = 'bic' 
output_file = f"arima-rolling-oneStep-{criterion}-L{train_points}.csv"

# ================== 读取数据与初始化 ==================
lines_df = pd.read_csv(input_file)
results_df = lines_df.copy()
for col in ['mae', 'mse', 'mape', 'r2', 'p', 'd', 'q', 'status']:
    results_df[col] = None

# ================== 循环处理每条线路 ==================
for idx, row in results_df.iterrows():
    line_name = row['line_name']
    print(f"\n[{idx+1}/{len(results_df)}] ARIMA 单步滚动测试: {line_name}")
    
    file_path = os.path.join(data_directory, line_name + '.csv')
    if not os.path.exists(file_path): continue
    
    try:
        # 1. 读取数据
        train, test = read_data_from_csv(
            file_name=file_path, column_name=column_name,
            train_points=train_points, forecast_horizon=forecast_horizon,
            sigma_threshold=3
        )
        
        # 2. 初始训练（寻找最优参数 p,d,q）
        pt = PowerTransformer(method='yeo-johnson')
        train_transformed = pt.fit_transform(train.reshape(-1, 1)).flatten()
        
        # 先找一次最优阶数
        stepwise_model = auto_arima(
            train_transformed, seasonal=True, m=12,
            information_criterion=criterion,
            suppress_warnings=True, error_action='ignore'
        )
        p, d, q = stepwise_model.order
        
        # 3. 【核心修改】单步滚动预测循环
        history = list(train_transformed)
        y_pred_list = []
        
        # 遍历测试集的 12 个点
        for t in range(len(test)):
            # 使用当前已知的 history 预测下一个点
            # 这种做法不用重训练模型，而是 update 数据，速度快且公平
            y_hat_transformed = stepwise_model.predict(n_periods=1)[0]
            
            # 反变换预测值并存起来
            y_hat = pt.inverse_transform(np.array([[y_hat_transformed]]))[0, 0]
            y_pred_list.append(y_hat)
            
            # 获取当前时刻的真实值，变换后喂给模型
            actual_transformed = pt.transform(np.array([[test[t]]]))[0, 0]
            stepwise_model.update([actual_transformed]) # 关键：模型吃到了真值
            
        y_pred = np.array(y_pred_list)
        y_true = test
        
        # 4. 指标计算
        mse_l = mean_squared_error(y_true, y_pred)
        mae_l = mean_absolute_error(y_true, y_pred)
        r2_l = r2_score(y_true, y_pred)
        mape_l = Mmape(y_true, y_pred)
        
        # 5. 记录
        results_df.loc[idx, ['mae', 'mse', 'mape', 'r2', 'p', 'd', 'q', 'status']] = \
            [mae_l, mse_l, mape_l, r2_l, p, d, q, 'success']
        
        print(f" 结果: R2={r2_l:.4f} | Mmape={mape_l:.4f} | Order({p},{d},{q})")
        
    except Exception as e:
        print(f" 出错: {str(e)}")
        results_df.loc[idx, 'status'] = f'failed: {str(e)}'

results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
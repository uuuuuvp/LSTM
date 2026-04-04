import os
import pandas as pd
import numpy as np
import torch
import warnings
import multiprocessing
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import Mmape, read_data_from_csv 

warnings.filterwarnings("ignore")

# ================== 参数设置 ==================
# 请确保路径和文件名正确
DATA_DIRECTORY = "E:/data/output_lines_0-6/"
# INPUT_FILE = "E:\\code\\mission\\full_sampled_lines.csv"
INPUT_FILE = "E:\Downloads\基于LSTM时间序列预测\基于LSTM时间序列预测\LSTM\Exp\LineIndex\storage.csv"
COLUMN_NAME = 'I_P'

# 实验对齐参数
TRAIN_POINTS = 1440        # 训练集点数（建议与LSTM保持一致，如1440）
FORECAST_HORIZON = 160     # 测试集总点数（滚动预测的总次数）
SCALES = [12, 24, 36, 84, 160] # 指标统计的尺度

# 模型参数
CRITERION = 'bic' 
OUTPUT_FILE = f"E:\\Downloads\\基于LSTM时间序列预测\\基于LSTM时间序列预测\\LSTM\Exp\\ARIMAResult\\arima-multiscale-results-{CRITERION}-{TRAIN_POINTS}-{COLUMN_NAME}.csv"
CHECKPOINT_FILE = f"checkpoint_arima_multiscale.csv"
SAVE_INTERVAL = 10 
WORKERS = 8 # 根据你的CPU核心数调整

# ================== 工具函数 ==================

def save_results(results_list, output_file, is_checkpoint=False):
    """保存结果到CSV，动态生成多尺度列名"""
    if not results_list:
        return
    
    df = pd.DataFrame(results_list)
    
    # 定义基础列
    base_cols = ['line_name', 'p', 'd', 'q', 'status']
    # 定义指标列
    metric_cols = []
    for s in SCALES:
        metric_cols.extend([f'mae_{s}', f'mse_{s}', f'mape_{s}', f'r2_{s}'])
    
    # 确保所有列在DataFrame中都存在（防止某些失败行缺失列）
    all_expected_cols = base_cols + metric_cols
    for col in all_expected_cols:
        if col not in df.columns:
            df[col] = None
            
    # 按照合理的顺序排列列
    final_cols = [c for c in all_expected_cols if c in df.columns]
    df = df[final_cols]
    
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    if is_checkpoint:
        print(f"   💾 检查点已保存: {len(results_list)} 条记录")

def load_checkpoint():
    """加载断点记录"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            checkpoint_df = pd.read_csv(CHECKPOINT_FILE)
            completed_lines = checkpoint_df['line_name'].dropna().tolist()
            print(f"🔄 发现检查点，已跳过 {len(completed_lines)} 条已完成线路")
            return checkpoint_df.to_dict('records'), set(completed_lines)
        except Exception as e:
            print(f"⚠️ 读取检查点失败: {e}")
    return [], set()

def process_single_line(line_info):
    """单条线路处理逻辑"""
    line_name, train_pts, f_horizon, crit, data_dir, col_name = line_info
    
    try:
        file_path = os.path.join(data_dir, f"{line_name}.csv")
        if not os.path.exists(file_path):
            return None
        
        # 1. 数据读取与切分
        # 假设 read_data_from_csv 返回 (train_np, test_np)
        train, test = read_data_from_csv(
            file_name=file_path, column_name=col_name,
            train_points=train_pts, forecast_horizon=f_horizon,
            sigma_threshold=3
        )
        
        if len(test) < max(SCALES):
            return {'line_name': line_name, 'status': f'failed: data_too_short_len_{len(test)}'}

        # 2. 数据变换 (PowerTransformer)
        pt = PowerTransformer(method='yeo-johnson')
        train_transformed = pt.fit_transform(train.reshape(-1, 1)).flatten()
        
        # 3. 自动寻找最佳 ARIMA 参数
        model = auto_arima(
            train_transformed, 
            seasonal=True, m=12,
            information_criterion=crit,
            suppress_warnings=True, 
            error_action='ignore'
        )
        p, d, q = model.order
        
        # 4. 单步滚动预测 160 次
        y_pred_list = []
        for t in range(len(test)):
            # 预测下一时刻
            y_hat_trans = model.predict(n_periods=1)[0]
            # 反变换回原始物理量级
            y_hat = pt.inverse_transform(np.array([[y_hat_trans]]))[0, 0]
            y_pred_list.append(y_hat)
            
            # 观测到真实值，更新模型（update 不会重新定阶，只是更新内部状态）
            actual_trans = pt.transform(np.array([[test[t]]]))[0, 0]
            model.update([actual_trans])
            
        y_pred = np.array(y_pred_list)
        y_true = test
        
        # 5. 构建结果字典
        line_res = {
            'line_name': line_name,
            'p': p, 'd': d, 'q': q,
            'status': 'success'
        }
        
        # 核心：多尺度计算
        for s in SCALES:
            s_true = y_true[:s]
            s_pred = y_pred[:s]
            
            line_res[f'mae_{s}'] = mean_absolute_error(s_true, s_pred)
            line_res[f'mse_{s}'] = mean_squared_error(s_true, s_pred)
            line_res[f'r2_{s}'] = r2_score(s_true, s_pred)
            line_res[f'mape_{s}'] = Mmape(s_true, s_pred)
            
        return line_res
        
    except Exception as e:
        return {'line_name': line_name, 'status': f'failed: {str(e)}'}

# ================== 主执行流程 ==================

if __name__ == "__main__":
    # 1. 准备线路
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        exit(1)
        
    lines_df = pd.read_csv(INPUT_FILE)
    all_line_names = lines_df['line_name'].tolist()
    
    # 2. 加载进度
    results_list, completed_set = load_checkpoint()
    pending_lines = [n for n in all_line_names if n not in completed_set]
    
    print(f"📊 总任务: {len(all_line_names)} | 已完成: {len(completed_set)} | 待处理: {len(pending_lines)}")
    
    if not pending_lines:
        print("✅ 所有任务已完成！")
        exit(0)

    # 3. 准备进程池参数
    process_args = [
        (name, TRAIN_POINTS, FORECAST_HORIZON, CRITERION, DATA_DIRECTORY, COLUMN_NAME)
        for name in pending_lines
    ]

    # 4. 并行执行
    print(f"🚀 启动并行计算 (Workers={WORKERS}, Scales={SCALES})...")
    
    try:
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            future_to_line = {executor.submit(process_single_line, args): args[0] for args in process_args}
            
            for idx, future in enumerate(as_completed(future_to_line), 1):
                line_name = future_to_line[future]
                try:
                    result = future.result()
                    if result:
                        results_list.append(result)
                    
                    # 定期保存进度
                    if len(results_list) % SAVE_INTERVAL == 0:
                        save_results(results_list, CHECKPOINT_FILE, is_checkpoint=True)
                    
                    print(f"[{len(results_list)}/{len(all_line_names)}] 完成: {line_name}")
                except Exception as e:
                    print(f"❌ 线程执行异常 {line_name}: {e}")
                    
    except KeyboardInterrupt:
        print("\n🛑 用户手动中断，正在保存当前进度...")
        save_results(results_list, CHECKPOINT_FILE, is_checkpoint=True)
        exit(1)

    # 5. 收尾保存
    save_results(results_list, OUTPUT_FILE, is_checkpoint=False)
    
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("🗑️ 清理检查点文件。")
        
    print(f"\n🎉 任务全部完成！结果已存至: {OUTPUT_FILE}")
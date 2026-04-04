import os
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import warnings
from utils import Mmape, read_data_from_csv 

warnings.filterwarnings("ignore")

# ================== 参数设置 ==================
data_directory = "E:/data/output_lines_0-6/"
input_file = "E:\\code\\mission\\full_sampled_lines.csv"
column_name = 'I_P'
train_points = 800
forecast_horizon = 12
criterion = 'bic' 
output_file = f"arima-rolling-oneStep-{criterion}.csv"
CHECKPOINT_FILE = f"checkpoint_{criterion}.csv"  # 临时检查点文件
SAVE_INTERVAL = 10  # 每10条保存一次

WORKERS = 8
print(f"使用 {WORKERS} 个进程并行处理")
print(f"每 {SAVE_INTERVAL} 条保存一次，支持断点续跑")

def save_results(results_list, output_file, is_checkpoint=False):
    """保存结果到CSV"""
    df = pd.DataFrame(results_list)
    for col in ['mae', 'mse', 'mape', 'r2', 'p', 'd', 'q', 'status']:
        if col not in df.columns:
            df[col] = None
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    if is_checkpoint:
        print(f"  💾 检查点已保存: {len(results_list)} 条 -> {output_file}")
    else:
        print(f"  💾 最终结果已保存: {len(results_list)} 条 -> {output_file}")

def load_checkpoint():
    """加载已有的检查点"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            checkpoint_df = pd.read_csv(CHECKPOINT_FILE)
            completed_lines = checkpoint_df['line_name'].tolist()
            print(f"🔄 发现已有检查点，已处理 {len(completed_lines)} 条线路")
            return checkpoint_df, set(completed_lines)
        except Exception as e:
            print(f"⚠️ 读取检查点失败: {e}，重新开始")
            return pd.DataFrame(), set()
    return pd.DataFrame(), set()

def process_single_line(line_info):
    """处理单条线路（完全保持原有逻辑）"""
    line_name, train_points, forecast_horizon, criterion, data_directory, column_name = line_info
    
    try:
        file_path = os.path.join(data_directory, line_name + '.csv')
        if not os.path.exists(file_path):
            return None
        
        train, test = read_data_from_csv(
            file_name=file_path, column_name=column_name,
            train_points=train_points, forecast_horizon=forecast_horizon,
            sigma_threshold=3
        )
        
        pt = PowerTransformer(method='yeo-johnson')
        train_transformed = pt.fit_transform(train.reshape(-1, 1)).flatten()
        
        stepwise_model = auto_arima(
            train_transformed, seasonal=True, m=12,
            information_criterion=criterion,
            suppress_warnings=True, error_action='ignore'
        )
        p, d, q = stepwise_model.order
        
        y_pred_list = []
        for t in range(len(test)):
            y_hat_transformed = stepwise_model.predict(n_periods=1)[0]
            y_hat = pt.inverse_transform(np.array([[y_hat_transformed]]))[0, 0]
            y_pred_list.append(y_hat)
            
            actual_transformed = pt.transform(np.array([[test[t]]]))[0, 0]
            stepwise_model.update([actual_transformed])
        
        y_pred = np.array(y_pred_list)
        y_true = test
        
        mse_l = mean_squared_error(y_true, y_pred)
        mae_l = mean_absolute_error(y_true, y_pred)
        r2_l = r2_score(y_true, y_pred)
        mape_l = Mmape(y_true, y_pred)
        
        return {
            'line_name': line_name,
            'mae': mae_l,
            'mse': mse_l,
            'mape': mape_l,
            'r2': r2_l,
            'p': p,
            'd': d,
            'q': q,
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'line_name': line_name,
            'mae': None,
            'mse': None,
            'mape': None,
            'r2': None,
            'p': None,
            'd': None,
            'q': None,
            'status': f'failed: {str(e)}'
        }

# ================== 主程序（支持断点续跑） ==================
if __name__ == "__main__":
    # 读取线路列表
    lines_df = pd.read_csv(input_file)
    all_line_names = lines_df['line_name'].tolist()
    
    # 加载检查点
    existing_results, completed_lines = load_checkpoint()
    
    # 找出未完成的线路
    pending_lines = [line for line in all_line_names if line not in completed_lines]
    
    print(f"总共需要处理 {len(all_line_names)} 条线路")
    print(f"已完成: {len(completed_lines)} 条")
    print(f"待处理: {len(pending_lines)} 条")
    
    if len(pending_lines) == 0:
        print("🎉 所有线路已处理完成！")
        exit(0)
    
    # 准备参数列表（只处理未完成的）
    process_args = [
        (line_name, train_points, forecast_horizon, criterion, data_directory, column_name)
        for line_name in pending_lines
    ]
    
    # 从已有结果开始
    if len(existing_results) > 0:
        results_list = existing_results.to_dict('records')
    else:
        results_list = []
    
    # 并行处理
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        future_to_line = {executor.submit(process_single_line, args): args[0] 
                         for args in process_args}
        
        completed_count = len(completed_lines)
        for idx, future in enumerate(as_completed(future_to_line), 1):
            line_name = future_to_line[future]
            try:
                result = future.result()
                if result is not None:
                    results_list.append(result)
                    print(f"[{completed_count + idx}/{len(all_line_names)}] 完成: {line_name} - success")
                else:
                    print(f"[{completed_count + idx}/{len(all_line_names)}] 跳过: {line_name} - 文件不存在")
                
                # 每 SAVE_INTERVAL 条保存检查点
                if len(results_list) % SAVE_INTERVAL == 0:
                    save_results(results_list, CHECKPOINT_FILE, is_checkpoint=True)
                    
            except Exception as e:
                print(f"[{completed_count + idx}/{len(all_line_names)}] 失败: {line_name} - {str(e)}")
                results_list.append({
                    'line_name': line_name,
                    'status': f'failed: {str(e)}',
                    'mae': None, 'mse': None, 'mape': None, 'r2': None,
                    'p': None, 'd': None, 'q': None
                })
    
    # 最终保存
    save_results(results_list, output_file, is_checkpoint=False)
    
    # 删除临时检查点文件
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"🗑️ 已删除临时检查点文件")
    
    # 统计
    success_count = len([r for r in results_list if r.get('status') == 'success'])
    print(f"\n✅ 全部处理完成！成功: {success_count}/{len(all_line_names)}")
    print(f"最终结果已保存至: {output_file}")
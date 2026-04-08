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

# ================== 配置 ==================
MODE = 'hour'  # 'hour' 或 'day'
TRAIN_POINTS = 1488
FORECAST_HORIZON = 168
SCALES = [12, 24, 36, 84, 168]
DAY_FLAG = False
M_VAL = 12

DATA_DIRECTORY = "E:/data/output_lines_0-6/"
INPUT_FILE = r"E:\Downloads\基于LSTM时间序列预测\基于LSTM时间序列预测\LSTM\Exp\LineIndex\full_sampled_lines.csv"
OUTPUT_FILE = "ARIMA_nor_bic_JQ.csv"
WORKERS = 6
SAVE_INTERVAL = 10
USE_YJ = True
CRITERION = 'bic'
COLUMN_NAME= 'J_Q'


# ================== 保存函数 ==================
def save_results(results_list, output_file, scales, is_checkpoint=False):
    if not results_list:
        return
    df = pd.DataFrame(results_list)
    base_cols = ['line_name', 'status', 'use_yj']
    metric_cols = []
    for s in scales:
        metric_cols.extend([f'nmae_{s}', f'nmse_{s}', f'mmape_{s}', f'r2_{s}'])
    all_expected_cols = base_cols + metric_cols
    for col in all_expected_cols:
        if col not in df.columns:
            df[col] = None
    df[all_expected_cols].to_csv(output_file, index=False, encoding='utf-8-sig')
    if is_checkpoint:
        print(f"   💾 检查点保存: {len(results_list)} 条")


# ================== 单条线路处理 ==================
def process_single_line_normalized(line_info, use_yj=True):
    line_name, train_pts, f_horizon, crit, data_dir, col_name, day_flag, m_val, scales = line_info
    try:
        file_path = os.path.join(data_dir, f"{line_name}.csv")
        if not os.path.exists(file_path):
            return None

        # 读取数据
        train, test = read_data_from_csv(
            file_name=file_path,
            column_name=col_name,
            train_points=train_pts,
            forecast_horizon=f_horizon,
            resample_to_day=day_flag
        )

        if len(test) < max(scales):
            return {'line_name': line_name, 'status': f'failed: len_{len(test)}_too_short'}

        # Yeo-Johnson 变换
        if use_yj:
            pt = PowerTransformer(method='yeo-johnson')
            train_transformed = pt.fit_transform(train.reshape(-1, 1)).flatten()
        else:
            train_transformed = train.copy()

        # 拟合 ARIMA
        model = auto_arima(
            train_transformed,
            seasonal=True,
            m=m_val,
            information_criterion=crit,
            suppress_warnings=True,
            error_action='ignore'
        )

        # 滚动预测
        y_pred_list = []
        for t in range(len(test)):
            y_hat_trans = model.predict(n_periods=1)[0]
            y_hat = pt.inverse_transform(np.array([[y_hat_trans]]))[0, 0] if use_yj else y_hat_trans
            y_pred_list.append(y_hat)

            # 更新模型
            actual_trans = pt.transform(np.array([[test[t]]]))[0, 0] if use_yj else test[t]
            model.update([actual_trans])

        y_pred = np.array(y_pred_list)
        y_true = test

        # 归一化指标
        scale_val = np.max(y_true) - np.min(y_true) + 1e-6
        line_res = {'line_name': line_name, 'status': 'success', 'use_yj': use_yj}

        for s in scales:
            s_true = y_true[:s]
            s_pred = y_pred[:s]

            nmae = mean_absolute_error(s_true, s_pred) / scale_val
            nmse = mean_squared_error(s_true, s_pred) / (scale_val ** 2)
            mmape = Mmape(s_true, s_pred)
            r2 = r2_score(s_true, s_pred)

            line_res[f'nmae_{s}'] = nmae
            line_res[f'nmse_{s}'] = nmse
            line_res[f'mmape_{s}'] = mmape
            line_res[f'r2_{s}'] = r2

        return line_res

    except Exception as e:
        return {'line_name': line_name, 'status': f'failed: {str(e)}'}


# ================== 批量执行 ==================
if __name__ == "__main__":
    lines_df = pd.read_csv(INPUT_FILE)
    all_line_names = lines_df['line_name'].tolist()

    process_args = [
        (name, TRAIN_POINTS, FORECAST_HORIZON, CRITERION,
         DATA_DIRECTORY, COLUMN_NAME, DAY_FLAG, M_VAL, SCALES)
        for name in all_line_names
    ]

    results_list = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        future_to_line = {
            executor.submit(process_single_line_normalized, args, USE_YJ): args[0]
            for args in process_args
        }
        for idx, future in enumerate(as_completed(future_to_line), 1):
            line_name = future_to_line[future]
            try:
                res = future.result()
                if res:
                    results_list.append(res)
                    if len(results_list) % SAVE_INTERVAL == 0:
                        save_results(results_list, "checkpoint_arima_normalized.csv", SCALES, is_checkpoint=True)
                print(f"[{idx}/{len(all_line_names)}] {line_name} - OK")
            except Exception as e:
                print(f"Error {line_name}: {e}")

    save_results(results_list, OUTPUT_FILE, SCALES)
    print(f"✅ 完成！结果路径: {OUTPUT_FILE}")
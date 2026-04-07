import pandas as pd
import torch, os, winsound, time, yaml
import numpy as np
from sklearn import preprocessing
from scipy import interpolate
from models import *
from utils import *
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

start = time.time()
# config_file = r"E:\Downloads\基于LSTM时间序列预测\基于LSTM时间序列预测\LSTM\Exp\ConfigPara\lstm_day_rolling.yaml"
config_file = r"E:\Downloads\基于LSTM时间序列预测\基于LSTM时间序列预测\LSTM\Exp\ConfigPara\lstm_hour_rolling.yaml"

with open(config_file, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# --- 配置解析 ---
day_flag = config['data_split']['day_flag']
sampling_method = config['data_split']['sampling_method']
train_ratio = config['data_split']['train_ratio']
val_ratio = config['data_split']['val_ratio']
test_ratio = config['data_split']['test_ratio']

batch_size = config['dataloader']['batch_size']
input_length = config['dataloader']['input_length']
output_length = config['dataloader']['output_length']
interval_length = config['dataloader']['interval_length']

epochs = config['training']['epochs']
# ======= 新增：Early Stop 开关 =======
# 优先从yaml读，如果没有则默认为True
early_stop_flag = config['training'].get('early_stop', False) 
diff_flag = config['training'].get('diff_flag', True)
# ====================================
loss_function = config['training']['loss_function']
learning_rate = config['training']['optimizer']['learning_rate']
weight_decay = config['training']['optimizer']['weight_decay']
loss_para = config['training']['loss_para']
clip_v = config['training']['clip_v']

dim = config['model']['dim']
num_blocks = config['model']['num_blocks']

scalar = config['preprocessing']['scalar']
scalar_contain_labels = config['preprocessing']['scalar_contain_labels']
target_value = config['preprocessing']['target_value']
outlier_flag = config['preprocessing']['outlier_flag']
interpolation = config['preprocessing']['interpolation']
features_num_config = config['preprocessing']['features_num']
threshold = config['preprocessing']['threshold']
Otl_Plt_M = config['preprocessing']['Otl_Plt_M']

data_directory = config['paths']['data_directory']
input_file = config['paths']['input_file']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pin_memory = (device.type == 'cuda')

lr_str = f"{learning_rate:.0e}".replace("e-0", "e")
exp_id = f"day_D{dim}B{num_blocks}_bs{batch_size}_{loss_function}{loss_para}_{Otl_Plt_M}{threshold}_E{epochs}_lr{lr_str}_f{features_num_config}_I{input_length}O{output_length}_C{clip_v}_rmT-MAX"
weight_dir = os.path.join("./weights", exp_id)

if os.path.exists(weight_dir):
    print(f"\n⏭️ [跳过] 实验 {exp_id} 已存在")
    import sys
    sys.exit(2)

os.makedirs(weight_dir, exist_ok=True)
output_file = f"E:\\Downloads\\基于LSTM时间序列预测\\基于LSTM时间序列预测\\LSTM\\Exp\\LSTMResult\\LSTM_{exp_id}.csv"

lines_df = pd.read_csv(input_file)
results_df = lines_df.copy()
for col in ['mae', 'mse', 'mape', 'Mmape', 'r2', 'status', 'train_loss', 'val_loss', 'loss_diff']:
    results_df[col] = None

# ================== 批处理循环 ==================
for idx, row in results_df.iterrows():
    line_name = row['line_name']
    save_path = os.path.join(weight_dir, f"{line_name}.pth")
    print(f"\n[{idx+1}/{len(results_df)}] 处理: {line_name}")

    file_name = os.path.join(data_directory, line_name + ".csv")
    if not os.path.exists(file_name):
        results_df.loc[idx, 'status'] = 'file_not_found'
        continue

    try:
        df = pd.read_csv(file_name)[:interval_length]
        if len(df) < 50:
            results_df.loc[idx, 'status'] = 'too_short'
            continue
        
        f_outlier(df, target_value, outlier_flag=outlier_flag, threshold=threshold, method=Otl_Plt_M)
        f_interpolation(df, target_value, interpolation=interpolation)
        
        if day_flag:
            df = resample_to_daily(df, target_value, method=sampling_method)
            print(f"天采样完成: {len(df)} 天数据，采样方法={sampling_method}")
        # 2. 处理时间格式（增加判断，避免天级别采样后报错）
        if 'timestamp' in df.columns:
            # 如果是采样后的 datetime 类型，就不再执行字符串替换和重新解析
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'].str.replace('国调_', ''), format='%Y%m%d_%H%M')
        # 3. 计算差分（要在采样之后算，才有“日增长”的意义）
        if diff_flag:
            df['diff_1'] = df[target_value].diff().fillna(0)
        # 4. 特征工程
        if features_num_config > 1:
            # 如果是天级别，hour 是固定的，sin/cos 会变成常数（全 0 或全 1），对模型没用
            if not day_flag:
                df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
            
            df['weekday_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.weekday / 7)
            df['weekday_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.weekday / 7)

        if day_flag:
            # 天级别模式：去掉小时特征，因为采样后小时是固定的
            if diff_flag:
                feature_cols = ['weekday_sin', 'weekday_cos', 'diff_1', target_value]
            else:
                feature_cols = ['weekday_sin', 'weekday_cos', target_value]
        else:
            # 小时级别模式：保留小时特征
            if features_num_config == 6 and diff_flag:
                feature_cols = ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'diff_1', target_value]
            elif features_num_config == 5:
                feature_cols = ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', target_value]
            elif features_num_config == 4 and diff_flag:
                feature_cols = ['hour_sin', 'hour_cos', 'diff_1', target_value]
            else:
                feature_cols = ['hour_sin', 'hour_cos', target_value]
        
        features_num = len(feature_cols)
        features_ = df[feature_cols].values
        labels_ = df[target_value].values

        # 数据切分与标准化
        split_train_val = int(len(features_) * train_ratio)
        if scalar:
            scalar_model = preprocessing.MinMaxScaler()
            train_f = scalar_model.fit_transform(features_[:split_train_val])
            val_test_f = scalar_model.transform(features_[split_train_val:])
            features_ = np.vstack([train_f, val_test_f])
            if scalar_contain_labels:
                labels_ = features_[:, -1]

        # 构造窗口
        features, labels = get_rolling_window_multistep(output_length, 0, input_length, features_.T, np.expand_dims(labels_, 0))
        labels = torch.squeeze(labels, dim=1).to(torch.float32)
        features = features.to(torch.float32)

        idx_train = int(len(features) * train_ratio)
        idx_val = idx_train + int(len(features) * val_ratio)

        train_Loader = DataLoader(TensorDataset(features[:idx_train], labels[:idx_train]), batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
        val_Loader = DataLoader(TensorDataset(features[idx_train:idx_val], labels[idx_train:idx_val]), batch_size=batch_size, pin_memory=pin_memory)
        test_Loader = DataLoader(TensorDataset(features[idx_val:], labels[idx_val:]), batch_size=1, pin_memory=pin_memory)

        # 模型初始化
        LSTMMain_model = LSTMMain(input_size=features_num, output_len=output_length, lstm_hidden=dim, lstm_layers=num_blocks, batch_size=batch_size, device=device).to(device)
        loss_func = nn.MSELoss() if loss_function == 'MSE' else nn.HuberLoss(delta=loss_para)
        optimizer = torch.optim.AdamW(LSTMMain_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-5)

        # --- 训练核心循环 ---
        print(f"Training Starts... (EarlyStop: {early_stop_flag})")
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        final_train_loss = 0
        final_val_loss = 0

        for epoch in range(epochs):
            LSTMMain_model.train()
            train_loss_sum = 0
            for feat, lb in train_Loader:
                feat, lb = feat.to(device), lb.to(device)
                optimizer.zero_grad()
                pred = LSTMMain_model(feat.permute(0, 2, 1))
                loss = loss_func(pred, lb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(LSTMMain_model.parameters(), clip_v)
                optimizer.step()
                train_loss_sum += loss.item()

            # 验证
            LSTMMain_model.eval()
            val_loss_sum = 0
            with torch.no_grad():
                for v_feat, v_lb in val_Loader:
                    v_pred = LSTMMain_model(v_feat.to(device).permute(0, 2, 1))
                    val_loss_sum += loss_func(v_pred, v_lb.to(device)).item()
            
            avg_train_loss = train_loss_sum / len(train_Loader)
            avg_val_loss = val_loss_sum / len(val_Loader)
            scheduler.step()

            # 权重保存逻辑
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(LSTMMain_model.state_dict(), save_path)
                patience_counter = 0
                final_train_loss, final_val_loss = avg_train_loss, avg_val_loss
            else:
                patience_counter += 1
                # ======= 修改点：仅在开关开启时触发 break =======
                if early_stop_flag and patience_counter >= patience:
                    print(f"Early Stopping at Epoch {epoch}")
                    break
                # =============================================
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # --- 测试阶段 ---
        LSTMMain_model.load_state_dict(torch.load(save_path))
        LSTMMain_model.eval()
        pre_array, label_array = [], []

        with torch.no_grad():
            for feat, lb in test_Loader:
                pred = LSTMMain_model(feat.to(device).permute(0, 2, 1))
                pre_array.append(pred.cpu().numpy())
                label_array.append(lb.cpu().numpy())

        pre_array = np.vstack(pre_array)
        label_array = np.vstack(label_array)

        # 反归一化处理
        if scalar_contain_labels and scalar:
            def inverse(data):
                return np.array([scalar_model.inverse_transform(np.concatenate([np.zeros((1, features_num-1)), d.reshape(1,-1)], axis=1))[0, -1] for d in data])
            pre_final = inverse(pre_array)
            label_final = inverse(label_array)
        else:
            pre_final, label_final = pre_array.flatten(), label_array.flatten()

        r2 = r2_score(label_final, pre_final)
        mape = mean_absolute_percentage_error(label_final, pre_final)
        print(f"结果: R2={r2:.4f} | MAPE={mape:.4f}")

        results_df.loc[idx, ['mae', 'mse', 'mape', 'Mmape', 'r2', 'status', 'train_loss', 'val_loss', 'loss_diff']] = [
            mean_absolute_error(label_final, pre_final), mean_squared_error(label_final, pre_final),
            mape, Mmape(label_final, pre_final), r2, 'success', final_train_loss, final_val_loss, final_val_loss - final_train_loss
        ]

    except Exception as e:
        print(f"失败: {e}")
        results_df.loc[idx, 'status'] = f'failed: {str(e)}'

    if (idx + 1) % 10 == 0:
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n✅ 任务完成！总耗时: {int((time.time()-start)//60)}分")
winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
os.startfile(output_file)
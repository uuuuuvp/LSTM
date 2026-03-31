import pandas as pd
import torch, os
from sklearn import preprocessing
from scipy import interpolate
from models import *
from utils import *
import yaml
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

config_file = "Exp/ConfigPara/lstm_hour.yaml"
with open(config_file, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

day_flag = config['data_split']['day_flag']
train_ratio = config['data_split']['train_ratio']
val_ratio = config['data_split']['val_ratio']
test_ratio = config['data_split']['test_ratio']

batch_size = config['dataloader']['batch_size']
input_length = config['dataloader']['input_length']
output_length = config['dataloader']['output_length']
interval_length = config['dataloader']['interval_length']

epochs = config['training']['epochs']
loss_function = config['training']['loss_function']
learning_rate = config['training']['optimizer']['learning_rate']
weight_decay = config['training']['optimizer']['weight_decay']

num_blocks = config['model']['num_blocks']
dim = config['model']['dim']

scalar = config['preprocessing']['scalar']
scalar_contain_labels = config['preprocessing']['scalar_contain_labels']
target_value = config['preprocessing']['target_value']
outlier_flag = config['preprocessing']['outlier_flag']
interpolation = config['preprocessing']['interpolation']
features_num = config['preprocessing']['features_num']
lag_points = config['preprocessing']['lag_points']
Otl_Plt_M = config['preprocessing']['Otl_Plt_M']
threshold = config['preprocessing']['threshold']
limit = config['preprocessing']['limit']

data_directory = config['paths']['data_directory']
input_file = config['paths']['input_file']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 老异常值检测，插值 命名
# output_file = f"./Exp/LSTMResult/LSTM-FulCzOt-{features_num}-{target_value}-{interval_length}-{input_length}-{output_length}.csv"
# 新异常值检测，插值一体化
output_file = f"./Exp/LSTMResult/FulOltPltMMAPE_ES-L{loss_function}E{epochs}B{batch_size}R{learning_rate}{Otl_Plt_M}-{threshold}-{features_num}-{target_value}L{interval_length}I{input_length}O{output_length}.csv"
# 测试
# output_file = f"./Exp/LSTMResult/cs_cz_{features_num}_{target_value}_{interval_length}_{input_length}_{output_length}.csv"

lines_df = pd.read_csv(input_file)

results_df = lines_df.copy()
results_df['mae'], results_df['mse'], results_df['mape'], results_df['r2'], results_df['status']= None, None, None, None, None
# results_df['mae'], results_df['mse'], results_df['mape'], results_df['safe_mape'], results_df['r2'], results_df['status']= None, None, None, None, None, None

if output_length > 1:
    forecasting_model = 'multi_steps'
else:
    forecasting_model = 'one_steps'

# ================== 批处理循环（新增） ==================
for idx, row in results_df.iterrows():
    line_name = row['line_name']
    print(f"\n[{idx+1}/{len(results_df)}] 处理: {line_name}")

    file_name = os.path.join(data_directory, line_name + ".csv")

    if not os.path.exists(file_name):
        print("文件不存在")
        results_df.loc[idx, 'status'] = 'file_not_found'
        continue

    try:
# ======================================================

        # ======== 原来的读取 ========
        df = pd.read_csv(file_name)
        df = df[:interval_length]

        if len(df) < 50:
            print("数据太短")
            results_df.loc[idx, 'status'] = 'too_short'
            continue
        
        # f_outlier(df, target_value=target_value, threshold=threshold, outlier_flag=outlier_flag)
        # f_interpolation(df, target_value=target_value, interpolation=interpolation)
        
        df, outlier_rate = Otl_Plt(df, target_col=target_value, method=Otl_Plt_M, threshold=threshold, lag=lag_points, limit=limit, outlier_flag=outlier_flag,interpolation=interpolation)
        
        if features_num > 1:
            df['timestamp'] = df['timestamp'].str.replace('国调_', '')
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d_%H%M')
            df['hour'] = df['timestamp'].dt.hour
            df['weekday'] = df['timestamp'].dt.weekday
            
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            if features_num == 5:
                df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
                df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
                feature_cols = ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', target_value]
            else:
                feature_cols = ['hour_sin', 'hour_cos', target_value]
            
            features_num = len(feature_cols) # 自动识别特征数，再次确认
            df_final = df[feature_cols]
        
        if features_num > 1:
            features_ = df_final.values
        else:
            features_ = df[target_value].values

        labels_ = df[target_value].values
        print(f"({min(labels_)})=>({max(labels_)})")

        split_train_val = int(len(features_)*train_ratio)
        split_val_test = split_train_val + int(len(features_)*val_ratio)

        # ======== 标准化（仅改一行：scalar重建）=======
        if scalar:
            train_features_ = features_[:split_train_val]
            val_test_features_ = features_[split_train_val:]

            scalar_model = preprocessing.MinMaxScaler()  # ✅ 防污染

            if features_num == 1:
                train_features_ = np.expand_dims(train_features_, axis=1)
                val_test_features_ = np.expand_dims(val_test_features_, axis=1)
            train_features_ = scalar_model.fit_transform(train_features_)
            val_test_features_ = scalar_model.transform(val_test_features_)
            # 重新将数据进行拼接
            features_ = np.vstack([train_features_, val_test_features_])
            if scalar_contain_labels:
                labels_ = features_[:, -1]

        if len(features_.shape) == 1:
            features_ = np.expand_dims(features_,0).T

        features, labels = get_rolling_window_multistep(
            output_length, 0, input_length,
            features_.T, np.expand_dims(labels_, 0)
        )

        labels = torch.squeeze(labels, dim=1)
        features = features.to(torch.float32)
        labels = labels.to(torch.float32)

        split_train_val = int(len(features)*train_ratio)
        split_val_test = split_train_val + int(len(features)*val_ratio)

        train_features = features[:split_train_val]
        train_labels = labels[:split_train_val]
        val_features = features[split_train_val:split_val_test]
        val_labels = labels[split_train_val:split_val_test]
        test_features = features[split_val_test:]
        test_labels = labels[split_val_test:]

        train_Datasets = TensorDataset(train_features.to(device), train_labels.to(device))
        train_Loader = DataLoader(batch_size=batch_size, dataset=train_Datasets)
        val_Datasets = TensorDataset(val_features.to(device), val_labels.to(device))
        val_Loader = DataLoader(batch_size=batch_size, dataset=val_Datasets)
        test_Datasets = TensorDataset(test_features.to(device), test_labels.to(device))
        test_Loader = DataLoader(batch_size=batch_size, dataset=test_Datasets)

        # ======== 模型（必须放进循环）=======
        LSTMMain_model = LSTMMain(
            input_size=features_num,
            output_len=output_length,
            lstm_hidden=dim,
            lstm_layers=num_blocks,
            batch_size=batch_size,
            device=device
        )

        LSTMMain_model.to(device)

        if loss_function == 'MSE':
            loss_func = nn.MSELoss(reduction='mean')
        elif loss_function == 'Huber':
            loss_func = nn.HuberLoss(delta=1.0)
        
        optimizer = torch.optim.AdamW(
            LSTMMain_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, epochs//3, eta_min=0.00001
        )
        
        patience =  10         # 如果连续7个epoch验证集损失不下降，则停止
        patience_counter = 0
        best_val_loss = float('inf')
        save_path = f'./weights/{line_name}_lstm.pth'
        
        print("Training Starts")
        for epoch in range(epochs):
            LSTMMain_model.train()
            train_loss_sum = 0

            for feature_, label_ in train_Loader:
                optimizer.zero_grad()
                feature_ = feature_.permute(0,2,1)
                prediction = LSTMMain_model(feature_)
                loss = loss_func(prediction, label_)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(LSTMMain_model.parameters(), 0.15)
                optimizer.step()
                train_loss_sum += loss.item()

            scheduler.step()
            
            # --- 验证阶段 (早停逻辑核心) ---
            LSTMMain_model.eval()
            val_loss_sum = 0
            with torch.no_grad():
                for val_feat, val_label in val_Loader:
                    val_feat = val_feat.permute(0, 2, 1)
                    val_pred = LSTMMain_model(val_feat)
                    v_loss = loss_func(val_pred, val_label)
                    val_loss_sum += v_loss.item()
            
            # 平均验证损失（按Batch平均）
            avg_val_loss = val_loss_sum / len(val_Loader)
            
# --- 早停判定 ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # 只有验证集表现更好时，才保存权重
                torch.save(LSTMMain_model.state_dict(), save_path)
                patience_counter = 0  # 重置计数器
                # print(f"Epoch {epoch}: Val loss improved to {best_val_loss:.6f}, saving model...")
            else:
                patience_counter += 1
                # print(f"Epoch {epoch}: Val loss did not improve. Counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}")
                break # 跳出当前线路的 epoch 循环
        
        # ======== 保存权重（防覆盖）=======
        torch.save(LSTMMain_model.state_dict(), save_path)

        # ======== 测试 ========
        LSTMMain_model.load_state_dict(torch.load(save_path))

        pre_array = None
        label_array = None

        for step, (feature_, label_) in enumerate(test_Loader):
            feature_ = feature_.permute(0, 2, 1)
            with torch.no_grad():
                prediction = LSTMMain_model(feature_)

            if step == 0:
                pre_array = prediction.cpu()
                label_array = label_.cpu()
            else:
                pre_array = np.vstack((pre_array, prediction.cpu()))
                label_array = np.vstack((label_array, label_.cpu()))

# ======== 反归一化（参照指定版本修改）========
        if scalar_contain_labels and scalar:
            pre_inverse = []
            test_inverse = []
            
            if features_num == 1 and output_length == 1:
                for pre_slice in range(pre_array.shape[0]):
                    pre_inverse_slice = scalar_model.inverse_transform(np.expand_dims(pre_array[pre_slice,:], axis=1))
                    test_inverse_slice = scalar_model.inverse_transform(np.expand_dims(label_array[pre_slice,:], axis=1))
                    pre_inverse.append(pre_inverse_slice)
                    test_inverse.append(test_inverse_slice)
                pre_array_final = np.array(pre_inverse).squeeze(axis=-1)
                test_labels_final = np.array(test_inverse).squeeze(axis=-1)
            
            elif features_num > 1:
                if isinstance(pre_array, np.ndarray):
                    pre_array = torch.from_numpy(pre_array)
                # 这里的 test_labels 是之前拆分出来的 Tensor
                for pre_slice in range(pre_array.shape[0]):
                    pre_inverse_slice = scalar_model.inverse_transform(torch.cat((torch.zeros(pre_array[0].shape[0], features_num-1), torch.unsqueeze(pre_array[pre_slice], dim=1)), 1))[:, -1]
                    test_inverse_slice = scalar_model.inverse_transform(torch.cat((torch.zeros(test_labels[0].shape[0], features_num-1), torch.unsqueeze(test_labels[pre_slice], dim=1)), 1))[:, -1]
                    pre_inverse.append(pre_inverse_slice)
                    test_inverse.append(test_inverse_slice)
                pre_array_final = np.array(pre_inverse)
                test_labels_final = np.array(test_inverse)
            
            else:
                for pre_slice in range(pre_array.shape[0]):
                    pre_inverse_slice = scalar_model.inverse_transform(np.expand_dims(pre_array[pre_slice,:], axis=1))
                    test_inverse_slice = scalar_model.inverse_transform(np.expand_dims(label_array[pre_slice,:], axis=1))
                    pre_inverse.append(pre_inverse_slice)
                    test_inverse.append(test_inverse_slice)
                pre_array_final = np.array(pre_inverse).squeeze(axis=-1)
                test_labels_final = np.array(test_inverse).squeeze(axis=-1)

        # ======== 指标计算（统一使用 _final 结尾的变量） ========
        MSE_l = mean_squared_error(test_labels_final, pre_array_final)
        MAE_l = mean_absolute_error(test_labels_final, pre_array_final)
        MAPE_l = mean_absolute_percentage_error(test_labels_final, pre_array_final)
        MMAPE_l = Mmape(test_labels_final, pre_array_final)
        R2 = r2_score(test_labels_final, pre_array_final)
        
        print(f"MSE={MSE_l:.2f} | MAE={MAE_l:.2f} | MAPE={MAPE_l:.2f} | safeMAPE={MMAPE_l:.2f} | R^2={R2:.2f}")
        # ======== 保存结果（新增）=======
        results_df.loc[idx, 'mae'] = MAE_l
        results_df.loc[idx, 'mse'] = MSE_l
        results_df.loc[idx, 'mape'] = MMAPE_l
        # results_df.loc[idx, 'safe_mape'] = safe_MAPE_l
        results_df.loc[idx, 'r2'] = R2
        results_df.loc[idx, 'status'] = 'success'

    except Exception as e:
        print("失败:", e)
        results_df.loc[idx, 'status'] = f'failed: {str(e)}'

    # ======== 每10条保存（新增）=======
    if (idx + 1) % 10 == 0:
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"已保存进度 {idx+1}")

# ================== 最终保存 ==================
results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print("全部完成")
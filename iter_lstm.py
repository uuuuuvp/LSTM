import pandas as pd
import torch, os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from models import *
from utils import *
import yaml
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

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

data_directory = config['paths']['data_directory']
input_file = config['paths']['input_file']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_file = f"./Exp/LSTMResult/LSTM_results_{target_value}_{interval_length}_{input_length}_{output_length}.csv"

lines_df = pd.read_csv(input_file)

results_df = lines_df.copy()
results_df['mae'] = None
results_df['mse'] = None
results_df['mape'] = None
results_df['r2'] = None
results_df['status'] = None

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

        features_num = 1
        if features_num > 1:
            features_ = df.values
        else:
            features_ = df[target_value].values

        labels_ = df[target_value].values
        print(max(labels_), min(labels_))

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

        
        optimizer = torch.optim.AdamW(
            LSTMMain_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, epochs//3, eta_min=0.00001
        )

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

        # ======== 保存权重（防覆盖）=======
        save_path = f'./weights/{line_name}_lstm.pth'
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

        # ======== 反归一化（只改scalar变量名）=======
        if scalar_contain_labels and scalar:
            pre_inverse = []
            test_inverse = []

            for i in range(pre_array.shape[0]):
                pre_inv = scalar_model.inverse_transform(
                    np.expand_dims(pre_array[i,:], axis=1)
                )
                test_inv = scalar_model.inverse_transform(
                    np.expand_dims(label_array[i,:], axis=1)
                )

                pre_inverse.append(pre_inv)
                test_inverse.append(test_inv)

            pre_array = np.array(pre_inverse).squeeze()
            test_labels_np = np.array(test_inverse).squeeze()

        else:
            test_labels_np = label_array.numpy()

        # ======== 指标 ========
        MSE_l = mean_squared_error(test_labels_np, pre_array)
        MAE_l = mean_absolute_error(test_labels_np, pre_array)
        MAPE_l = mean_absolute_percentage_error(test_labels_np, pre_array)
        R2 = r2_score(test_labels_np, pre_array)

        print(f"成功: MAE={MAE_l:.4f}, MAPE={MAPE_l:.4f}")

        # ======== 保存结果（新增）=======
        results_df.loc[idx, 'mae'] = MAE_l
        results_df.loc[idx, 'mse'] = MSE_l
        results_df.loc[idx, 'mape'] = MAPE_l
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
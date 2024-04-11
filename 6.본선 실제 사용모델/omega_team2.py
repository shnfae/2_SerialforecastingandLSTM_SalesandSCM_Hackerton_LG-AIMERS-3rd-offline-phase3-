import random
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import copy

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'TRAIN_WINDOW_SIZE':60,
    'PREDICT_SIZE':21,
    'EPOCHS':5,
    'LEARNING_RATE':1e-4,
    'BATCH_SIZE':1024,
    'SEED':41
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(CFG['SEED']) # Seed 고정


### feature 3개 + 1 drop일경우

train_data = pd.read_csv('/data/user/10.omega/temp_output/train_b_new.csv').drop(columns=['ID', '제품'])
train_data_2 = pd.read_csv('/data/user/10.omega/temp_output/train_b_new.csv').drop(columns=[ '제품'])

indexs_bigcat={} ## psfa score용
for bigcat in train_data['대분류'].unique():
    indexs_bigcat[bigcat] = list(train_data.loc[train_data['대분류']==bigcat].index)
indexs_bigcat.keys()

numeric_cols = train_data.columns[5:]
numeric_cols_2 = train_data_2.columns[6:]

train_data[numeric_cols] = train_data[numeric_cols].apply(pd.to_numeric, errors='coerce')
train_data_2[numeric_cols_2] = train_data_2[numeric_cols_2].apply(pd.to_numeric, errors='coerce')



# 칵 column의 min 및 max 계산
min_values = train_data[numeric_cols].min(axis=1)
max_values = train_data[numeric_cols].max(axis=1)
# 각 행의 범위(max-min)를 계산하고, 범위가 0인 경우 1로 대체
ranges = max_values - min_values
ranges[ranges == 0] = 1
# min-max scaling 수행
train_data[numeric_cols] = train_data[numeric_cols].subtract(min_values, axis=0).div(ranges, axis=0)
# max와 min 값을 dictionary 형태로 저장
scale_min_dict = min_values.to_dict()
scale_max_dict = max_values.to_dict()

print("success")



# 칵 column의 min 및 max 계산
min_values_2 = train_data_2[numeric_cols_2].min(axis=1)
max_values_2 = train_data_2[numeric_cols_2].max(axis=1)
# 각 행의 범위(max-min)를 계산하고, 범위가 0인 경우 1로 대체
ranges_2 = max_values_2 - min_values_2
ranges_2[ranges_2 == 0] = 1
# min-max scaling 수행
train_data_2[numeric_cols_2] = train_data_2[numeric_cols_2].subtract(min_values_2, axis=0).div(ranges_2, axis=0)
# max와 min 값을 dictionary 형태로 저장
scale_min_dict_2 = min_values_2.to_dict()
scale_max_dict_2 = max_values_2.to_dict()

# 칵 column의 min 및 max 계산


label_encoder = LabelEncoder()
categorical_columns = ['대분류', '중분류', '소분류', '브랜드','쇼핑몰']

for col in categorical_columns:
    label_encoder.fit(train_data[col])
    train_data[col] = label_encoder.transform(train_data[col])
label_encoder_bigcat = LabelEncoder()
train_data_2['대분류_encoded'] = label_encoder_bigcat.fit_transform(train_data_2['대분류'])

label_encoder_id = LabelEncoder()
train_data_2['중분류_encoded'] = label_encoder_id.fit_transform(train_data_2['중분류'])
train_data_2['소분류_encoded'] = label_encoder_id.fit_transform(train_data_2['소분류'])
train_data_2['브랜드_encoded'] = label_encoder_id.fit_transform(train_data_2['브랜드'])
train_data_2['쇼핑몰_encoded'] = label_encoder_id.fit_transform(train_data_2['쇼핑몰'])
train_data_2['ID_encoded'] = label_encoder_id.fit_transform(train_data_2['ID'])


def make_train_data(data,train_size=CFG['TRAIN_WINDOW_SIZE'], predict_size=CFG['PREDICT_SIZE']):
    num_rows = len(data)
    window_size = train_size + predict_size
    input_data = np.empty((num_rows * (len(data.columns) - window_size + 1), train_size, len(data.iloc[0, :5]) + 1)) ## 여긴 분류지우는거랑 상관 없음
    target_data = np.empty((num_rows * (len(data.columns) - window_size + 1), predict_size))
    for i in tqdm(range(num_rows)):
        encode_info = np.array(data.iloc[i, :5])
        sales_data = np.array(data.iloc[i, 5:])

        for j in range(len(sales_data) - window_size + 1):
            window_sales = sales_data[j : j + window_size]
            temp_data = np.column_stack((np.tile(encode_info, (train_size, 1)), window_sales[:train_size]))
            input_data[i * (len(data.columns) - window_size + 1) + j] = temp_data
            target_data[i * (len(data.columns) - window_size + 1) + j] = window_sales[train_size:]
    return input_data, target_data

def make_predict_data(data,train_size=CFG['TRAIN_WINDOW_SIZE']):
    num_rows = len(data)
    input_data = np.empty((num_rows, train_size, len(data.iloc[0, :5]) + 1))
    for i in tqdm(range(num_rows)):
        encode_info = np.array(data.iloc[i, :5])
        sales_data = np.array(data.iloc[i, -train_size:])
        window_sales = sales_data[-train_size : ]
        temp_data = np.column_stack((np.tile(encode_info, (train_size, 1)), window_sales[:train_size]))
        input_data[i] = temp_data
    return input_data

def create_new_val_loader_modified(data, train_size=CFG['TRAIN_WINDOW_SIZE'], predict_size=CFG['PREDICT_SIZE'], feature_size=6):
    num_rows = len(data)
    input_data = np.empty((num_rows, train_size, feature_size))
    target_data = np.empty((num_rows, predict_size))
    for i, (_, row) in enumerate(data.iterrows()):
        encode_info = np.array([row['대분류_encoded'],row['중분류_encoded'], row['소분류_encoded'], row['브랜드_encoded'],row['쇼핑몰_encoded']])
        sales_data = np.array(row[-(train_size + predict_size):])
        if len(sales_data) < train_size + predict_size:
            print(f"Row {i} has insufficient data. Skipping...")
            continue
        window_sales = sales_data[:train_size]
        temp_data = np.column_stack((np.tile(encode_info, (train_size, 1)), window_sales))
        input_data[i] = temp_data
        target_data[i] = sales_data[train_size:]
    dataset = CustomDataset_2(input_data, target_data)
    loader = DataLoader(dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
    return loader



train_input, train_target = make_train_data(train_data)
test_input = make_predict_data(train_data)

data_len = len(train_input)
val_input = train_input[-int(data_len*0.2):]
val_target = train_target[-int(data_len*0.2):]
train_input = train_input[:-int(data_len*0.2)]
train_target = train_target[:-int(data_len*0.2)]
train_input.shape, train_target.shape, val_input.shape, val_target.shape, test_input.shape

# 5. dataset 정의 및 만들기
class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __getitem__(self, index):
        if self.Y is not None:
            return torch.Tensor(self.X[index]), torch.Tensor(self.Y[index])
        return torch.Tensor(self.X[index])
    def __len__(self):
        return len(self.X)
class CustomDataset_2():
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data
    def __len__(self):
        return len(self.input_data)
    def __getitem__(self, idx):
        if self.target_data is not None:
            return self.input_data[idx], self.target_data[idx]
        return self.input_data[idx],

train_dataset = CustomDataset(train_input, train_target)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0 )

val_dataset = CustomDataset(val_input, val_target)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
new_val_loader = create_new_val_loader_modified(train_data_2)




class LTSF_NLinear(torch.nn.Module):
    def __init__(self, window_size=CFG['TRAIN_WINDOW_SIZE'], forcast_size=CFG['PREDICT_SIZE'], individual=True, feature_size=9):
        super(LTSF_NLinear, self).__init__()
        self.window_size = window_size
        self.forcast_size = forcast_size
        self.individual = individual
        self.channels = feature_size

        if self.individual:
            self.Linear = torch.nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(torch.nn.Linear(self.window_size, self.forcast_size))
        else:
            self.Linear = torch.nn.Linear(self.window_size, self.forcast_size)

        self.final_layer = torch.nn.Linear(self.channels, 1)  # To select only sales

    def forward(self, x):
        x=x.float()
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last

        if self.individual:
            output = torch.zeros([x.size(0), self.forcast_size, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)

        # This will select only the sales (assuming it's the first feature)
        x = self.final_layer(x)
        adjusted_seq_last = seq_last[:,:,2].unsqueeze(-1).expand(-1, 21, -1)
        x = x + adjusted_seq_last

        return x.squeeze(-1)  # Output shape [1024, 21]
class informerlstm(nn.Module):
    def __init__(self, input_size=6, hidden_size=512, output_size=CFG['PREDICT_SIZE']):
        super(informerlstm, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True,bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, output_size)
        )
        self.relu = nn.ReLU()

        self.informer = LTSF_NLinear(window_size=CFG['TRAIN_WINDOW_SIZE'], forcast_size=output_size, individual=True, feature_size=input_size)

    def forward(self, x):
        x = x.float()
        h_0 = torch.zeros(2, x.size(0), self.hidden_size, device=x.device)
        c_0 = torch.zeros(2, x.size(0), self.hidden_size, device=x.device)

        lstm_out, _ = self.lstm(x, (h_0, c_0))
        last_output = lstm_out[:, -1, :]
        x = self.informer(x)  # Note: Make sure the shapes are compatible
        x = self.relu(self.fc(last_output))

        return x  # Output shape [batch_size, output_size]
def train(model, optimizer, train_loader, val_loader,new_val_loader, device):
    model.to(device)
    criterion = nn.MSELoss().to(device)
    best_loss = 9999999
    best_model = None
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        train_mae = []
        for X, Y in tqdm(iter(train_loader)):
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, Y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        val_loss = validation(model, val_loader, criterion, device)
        print(f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}]')
        if best_loss > val_loss:
            best_loss = val_loss
            best_model = model
            print('Model Saved')
        model_copy = copy.deepcopy(model)
        psfa_score = validation_2(model_copy, new_val_loader, device)
        print(psfa_score)

    return best_model

def validation(model, val_loader, criterion, device):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for X, Y in tqdm(iter(val_loader)):
            X = X.to(device)
            Y = Y.to(device)
            output = model(X)
            loss = criterion(output, Y)
            val_loss.append(loss.item())
    return np.mean(val_loss)

def validation_2(model, new_val_loader, device):
    model.eval()
    pred_val_tensors = []
    target_val_tensors = []

    with torch.no_grad():
        for X, Y in iter(new_val_loader):
            X = X.to(device)
            Y = Y.to(device)
            
            output_val = model(X)
            pred_val_tensors.append(output_val.cpu())
            target_val_tensors.append(Y.cpu())

    pred_val = torch.cat(pred_val_tensors, dim=0).numpy()
    target_val = torch.cat(target_val_tensors, dim=0).numpy()

    # Initializing the PSFA scores
    psfa_m_scores = np.zeros(5)  # Assuming 5 categories as per your description
    psfa_m_scores = {}

    for cat in indexs_bigcat.keys():  # Iterate over each category
        daily_scores = []
        ids = indexs_bigcat[cat]  # Getting the product indices for the category
        
        for day in range(21):  # Iterate over each day
            total_sell = np.sum(target_val[ids, day])
            pred_values = pred_val[ids, day]
            target_values = target_val[ids, day]
            
            denominator = np.maximum(target_values, pred_values)
            numerators = np.abs(target_values - pred_values)

            with np.errstate(divide='ignore', invalid='ignore'):
                diffs = np.where(denominator != 0, numerators / denominator, 0)
            diffs[np.isnan(diffs)] = 0

            if total_sell != 0:
                sell_weights = target_values / total_sell
            else:
                sell_weights = np.ones_like(target_values) / len(ids)

            daily_scores.append(np.sum(diffs * sell_weights))

        psfa_m_scores[cat] = 1 - np.mean(daily_scores)

    # Compute the final PSFA score
    psfa_score = np.mean(list(psfa_m_scores.values()))
    return psfa_score

# 7. 모델학습하기
model = informerlstm()
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
infer_model = train(model, optimizer, train_loader, val_loader,new_val_loader, device)

test_dataset = CustomDataset(test_input, None)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


def inference(infer_model, test_loader, device):
    predictions = []
    with torch.no_grad():
        for X in tqdm(iter(test_loader)):
            X = X.to(device)
            output = infer_model(X)
            # 모델 출력인 output을 CPU로 이동하고 numpy 배열로 변환
            output = output.cpu().numpy()
            predictions.extend(output)
    return np.array(predictions)
pred = inference(infer_model, test_loader, device)

# 9. 후처리 및 저장
for idx in range(len(pred)):
    pred[idx, :] = pred[idx, :] * (scale_max_dict[idx] - scale_min_dict[idx]) + scale_min_dict[idx]
pred = np.round(pred, 0).astype(int)
pred.shape

model_path = '/data/user/10.omega/output/model_train_3.pth'
torch.save(infer_model.state_dict(), model_path)
submit = pd.read_csv('/data/user/10.omega/temp_output/sample_submission_b_new.csv')
submit.head()
submit.iloc[:,1:] = pred
submit.head()
submit.to_csv('/data/user/10.omega/type_output/baseline_submit_omega_new_b_seraph_60.csv', index=False)

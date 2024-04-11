train_data = pd.read_csv('/data/user/postreview/temp_output/train_1.csv').drop(columns=['ID', '제품'])
train_data_2 = pd.read_csv('/data/user/postreview/temp_output/train_1.csv').drop(columns=['제품'])
train_feature1_data = pd.read_csv('/data/user/postreview/temp_output/brand_1.csv').drop(columns=['ID', '제품'])
train_feature1_data_2 = pd.read_csv('/data/user/postreview/temp_output/brand_1.csv').drop(columns=['제품'])
train_feature2_data = pd.read_csv('/data/user/postreview/temp_output/weekday_1.csv').drop(columns=['ID', '제품'])
train_feature2_data_2 = pd.read_csv('/data/user/postreview/temp_output/weekday_1.csv').drop(columns=['제품'])
train_feature3_data = pd.read_csv('/data/user/postreview/temp_output/sales_price_1.csv').drop(columns=['ID', '제품'])
train_feature3_data_2 = pd.read_csv('/data/user/postreview/temp_output/sales_price_1.csv').drop(columns=['제품'])

# 2. 라벨링 및 스케일링
indexs_bigcat={}
for bigcat in train_data['대분류'].unique():
    indexs_bigcat[bigcat] = list(train_data.loc[train_data['대분류']==bigcat].index)
indexs_bigcat.keys()

scale_max_dict = {}
scale_min_dict = {}
for idx in tqdm(range(len(train_data))):
    maxi = np.max(train_data.iloc[idx,4:])
    mini = np.min(train_data.iloc[idx,4:])
    if maxi == mini :
        train_data.iloc[idx,4:] = 0
    else:
        train_data.iloc[idx,4:] = (train_data.iloc[idx,4:] - mini) / (maxi - mini)
    scale_max_dict[idx] = maxi
    scale_min_dict[idx] = mini

scale_max_dict_2 = {}
scale_min_dict_2 = {}
for idx in tqdm(range(len(train_data_2))):
    maxi = np.max(train_data_2.iloc[idx,5:])
    mini = np.min(train_data_2.iloc[idx,5:])
    if maxi == mini :
        train_data_2.iloc[idx,5:] = 0
    else:
        train_data_2.iloc[idx,5:] = (train_data_2.iloc[idx,5:] - mini) / (maxi - mini)
    scale_max_dict_2[idx] = maxi
    scale_min_dict_2[idx] = mini

scale_max_dict_features_1 = {}
scale_min_dict_features_1 = {}
for idx in tqdm(range(len(train_feature1_data))):
    maxi = np.max(train_feature1_data.iloc[idx,4:])
    mini = np.min(train_feature1_data.iloc[idx,4:])
    if maxi == mini :
        train_feature1_data.iloc[idx,4:] = 0
    else:
        train_feature1_data.iloc[idx,4:] = (train_feature1_data.iloc[idx,4:] - mini) / (maxi - mini)
    scale_max_dict_features_1[idx] = maxi
    scale_min_dict_features_1[idx] = mini

scale_max_dict_features_1_2 = {}
scale_min_dict_features_1_2 = {}
for idx in tqdm(range(len(train_feature1_data_2))):
    maxi = np.max(train_feature1_data_2.iloc[idx,5:])
    mini = np.min(train_feature1_data_2.iloc[idx,5:])
    if maxi == mini :
        train_feature1_data_2.iloc[idx,5:] = 0
    else:
        train_feature1_data_2.iloc[idx,5:] = (train_feature1_data_2.iloc[idx,5:] - mini) / (maxi - mini)
    scale_max_dict_features_1_2[idx] = maxi
    scale_min_dict_features_1_2[idx] = mini

scale_max_dict_feature2 = {}
scale_min_dict_feature2 = {}
for idx in tqdm(range(len(train_feature2_data))):
    maxi = np.max(train_feature2_data.iloc[idx,4:])
    mini = np.min(train_feature2_data.iloc[idx,4:])
    if maxi == mini :
        train_feature2_data.iloc[idx,4:] = 0
    else:
        train_feature2_data.iloc[idx,4:] = (train_feature2_data.iloc[idx,4:] - mini) / (maxi - mini)
    scale_max_dict_feature2[idx] = maxi
    scale_min_dict_feature2[idx] = mini

scale_max_dict_feature2_2 = {}
scale_min_dict_feature2_2 = {}
for idx in tqdm(range(len(train_feature2_data_2))):
    maxi = np.max(train_feature2_data_2.iloc[idx,5:])
    mini = np.min(train_feature2_data_2.iloc[idx,5:])
    if maxi == mini :
        train_feature2_data_2.iloc[idx,5:] = 0
    else:
        train_feature2_data_2.iloc[idx,5:] = (train_feature2_data_2.iloc[idx,5:] - mini) / (maxi - mini)
    scale_max_dict_feature2_2[idx] = maxi
    scale_min_dict_feature2_2[idx] = mini

scale_max_dict_feature3 = {}
scale_min_dict_feature3 = {}
for idx in tqdm(range(len(train_feature3_data))):
    maxi = np.max(train_feature3_data.iloc[idx,4:])
    mini = np.min(train_feature3_data.iloc[idx,4:])
    if maxi == mini :
        train_feature3_data.iloc[idx,4:] = 0
    else:
        train_feature3_data.iloc[idx,4:] = (train_feature3_data.iloc[idx,4:] - mini) / (maxi - mini)
    scale_max_dict_feature3[idx] = maxi
    scale_min_dict_feature3[idx] = mini

scale_max_dict_feature3_2 = {}
scale_min_dict_feature3_2 = {}
for idx in tqdm(range(len(train_feature3_data_2))):
    maxi = np.max(train_feature3_data_2.iloc[idx,5:])
    mini = np.min(train_feature3_data_2.iloc[idx,5:])
    if maxi == mini :
        train_feature3_data_2.iloc[idx,5:] = 0
    else:
        train_feature3_data_2.iloc[idx,5:] = (train_feature3_data_2.iloc[idx,5:] - mini) / (maxi - mini)
    scale_max_dict_feature3_2[idx] = maxi
    scale_min_dict_feature3_2[idx] = mini



label_encoder = LabelEncoder()
categorical_columns = ['대분류', '중분류', '소분류', '브랜드']

for col in categorical_columns:
    label_encoder.fit(train_data[col])
    train_data[col] = label_encoder.transform(train_data[col])
label_encoder_bigcat = LabelEncoder()
train_data_2['대분류_encoded'] = label_encoder_bigcat.fit_transform(train_data_2['대분류'])

label_encoder_id = LabelEncoder()
train_data_2['중분류_encoded'] = label_encoder_id.fit_transform(train_data_2['중분류'])
train_data_2['소분류_encoded'] = label_encoder_id.fit_transform(train_data_2['소분류'])
train_data_2['브랜드_encoded'] = label_encoder_id.fit_transform(train_data_2['브랜드'])
train_data_2['ID_encoded'] = label_encoder_id.fit_transform(train_data_2['ID'])




def make_train_data(data,data2,data3,data4, train_size=CFG['TRAIN_WINDOW_SIZE'], predict_size=CFG['PREDICT_SIZE']):
    num_rows = len(data)
    window_size = train_size + predict_size
    input_data = np.empty((num_rows * (len(data.columns) - window_size + 1), train_size, len(data.iloc[0, :4]) + 4))
    target_data = np.empty((num_rows * (len(data.columns) - window_size + 1), predict_size))
    for i in tqdm(range(num_rows)):
        encode_info = np.array(data.iloc[i, :4])
        sales_data = np.array(data.iloc[i, 4:])
        feature1_data = np.array(data2.iloc[i, 4:])
        feature2_data = np.array(data3.iloc[i, 4:])
        feature3_data = np.array(data4.iloc[i, 4:])

        for j in range(len(sales_data) - window_size + 1):
            window_sales = sales_data[j : j + window_size]
            window_feature1 = feature1_data[j : j + window_size]
            window_feature2 = feature2_data[j : j + window_size]
            window_feature3 = feature3_data[j : j + window_size]
            temp_data = np.column_stack((np.tile(encode_info, (train_size, 1)), window_sales[:train_size], window_feature1[:train_size],window_feature2[:train_size],window_feature3[:train_size]))
            input_data[i * (len(data.columns) - window_size + 1) + j] = temp_data
            target_data[i * (len(data.columns) - window_size + 1) + j] = window_sales[train_size:]
    return input_data, target_data

def make_predict_data(data, data2,data3,data4, train_size=CFG['TRAIN_WINDOW_SIZE']):
    num_rows = len(data)
    input_data = np.empty((num_rows, train_size, len(data.iloc[0, :4]) + 4))
    for i in tqdm(range(num_rows)):
        encode_info = np.array(data.iloc[i, :4])
        sales_data = np.array(data.iloc[i, -train_size:])
        feature1_data = np.array(data2.iloc[i, -train_size:])
        feature2_data = np.array(data3.iloc[i, -train_size:])
        feature3_data = np.array(data4.iloc[i, -train_size:])

        window_sales = sales_data[-train_size : ]
        window_feature1 = feature1_data[-train_size : ]
        window_feature2 = feature2_data[-train_size : ]
        window_feature3 = feature3_data[-train_size : ]

        temp_data = np.column_stack((np.tile(encode_info, (train_size, 1)), window_sales[:train_size], window_feature1[:train_size],window_feature2[:train_size],window_feature3[:train_size]))
        input_data[i] = temp_data
    return input_data

def create_new_val_loader_modified(data,data2,data3,data4, train_size=CFG['TRAIN_WINDOW_SIZE'], predict_size=CFG['PREDICT_SIZE'], feature_size=8):
    num_rows = len(data)
    input_data = np.empty((num_rows, train_size, feature_size))
    target_data = np.empty((num_rows, predict_size))
    for i, (_, row) in enumerate(data.iterrows()):
        encode_info = np.array([row['대분류_encoded'], row['중분류_encoded'], row['소분류_encoded'], row['브랜드_encoded']])
        sales_data = np.array(row[-(train_size + predict_size):])
        feature1_data = np.array(data2.iloc[i, -(train_size + predict_size):])
        feature2_data = np.array(data3.iloc[i, -(train_size + predict_size):])
        feature3_data = np.array(data4.iloc[i, -(train_size + predict_size):])
        if len(sales_data) < train_size + predict_size:
            print(f"Row {i} has insufficient data. Skipping...")
            continue
        window_sales = sales_data[:train_size]
        window_feature1 = feature1_data[:train_size]
        window_feature2 = feature2_data[:train_size]
        window_feature3 = feature3_data[:train_size]
        temp_data = np.column_stack((np.tile(encode_info, (train_size, 1)), window_sales, window_feature1,window_feature2,window_feature3))
        input_data[i] = temp_data
        target_data[i] = sales_data[train_size:]
    dataset = CustomDataset_2(input_data, target_data)
    loader = DataLoader(dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
    return loader



# 4. data 만들기
train_input, train_target = make_train_data(train_data,train_feature1_data,train_feature2_data,train_feature3_data)
test_input = make_predict_data(train_data,train_feature1_data,train_feature2_data,train_feature3_data)

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
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = CustomDataset(val_input, val_target)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
new_val_loader = create_new_val_loader_modified(train_data_2,train_feature1_data_2,train_feature2_data_2,train_feature3_data_2)


# 1. 데이터 불러오기
train_data = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/temp_output/train_2.csv').drop(columns=['ID', '제품','중분류'])
train_data_2 = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/temp_output/train_2.csv').drop(columns=[ '제품','중분류'])
train_data_features_1_1 = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/temp_output/brand_2_1_327.csv').drop(columns=['ID', '제품','중분류'])
train_data_features_1_2 = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/temp_output/brand_2_1_327.csv').drop(columns=[ '제품','중분류'])
train_data_features_2_1 = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/temp_output/weekday_2_1_327.csv').drop(columns=['ID', '제품','중분류'])
train_data_features_2_2 = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/temp_output/weekday_2_1_327.csv').drop(columns=[ '제품','중분류'])
train_data_features_3_1 = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/temp_output/sales_price_2_1_327.csv').drop(columns=['ID', '제품','중분류'])
train_data_features_3_2 = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/temp_output/sales_price_2_1_327.csv').drop(columns=[ '제품','중분류'])

indexs_bigcat={} ## psfa score용
for bigcat in train_data['대분류'].unique():
    indexs_bigcat[bigcat] = list(train_data.loc[train_data['대분류']==bigcat].index)
indexs_bigcat.keys()

numeric_cols = train_data.columns[3:]
numeric_cols_2 = train_data_2.columns[4:]
numeric_cols_1_1 = train_data_features_1_1.columns[3:]
numeric_cols_1_2 = train_data_features_1_2.columns[4:]
numeric_cols_2_1 = train_data_features_2_1.columns[3:]
numeric_cols_2_2 = train_data_features_2_2.columns[4:]
numeric_cols_3_1 = train_data_features_3_1.columns[3:]
numeric_cols_3_2 = train_data_features_3_2.columns[4:]

train_data[numeric_cols] = train_data[numeric_cols].apply(pd.to_numeric, errors='coerce')
train_data_2[numeric_cols_2] = train_data_2[numeric_cols_2].apply(pd.to_numeric, errors='coerce')
train_data_features_1_1[numeric_cols_1_1] = train_data_features_1_1[numeric_cols_1_1].apply(pd.to_numeric, errors='coerce')
train_data_features_1_2[numeric_cols_1_2] = train_data_features_1_2[numeric_cols_1_2].apply(pd.to_numeric, errors='coerce')
train_data_features_2_1[numeric_cols_2_1] = train_data_features_2_1[numeric_cols_2_1].apply(pd.to_numeric, errors='coerce')
train_data_features_2_2[numeric_cols_2_2] = train_data_features_2_2[numeric_cols_2_2].apply(pd.to_numeric, errors='coerce')
train_data_features_3_1[numeric_cols_3_1] = train_data_features_3_1[numeric_cols_3_1].apply(pd.to_numeric, errors='coerce')
train_data_features_3_2[numeric_cols_3_2] = train_data_features_3_2[numeric_cols_3_2].apply(pd.to_numeric, errors='coerce')



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
min_values_features_1_1 = train_data_features_1_1[numeric_cols_1_1].min(axis=1)
max_values_features_1_1 = train_data_features_1_1[numeric_cols_1_1].max(axis=1)
# 각 행의 범위(max-min)를 계산하고, 범위가 0인 경우 1로 대체
ranges_features_1_1 = max_values_features_1_1 - min_values_features_1_1
ranges_features_1_1[ranges_features_1_1 == 0] = 1
# min-max scaling 수행
train_data_features_1_1[numeric_cols_1_1] = train_data_features_1_1[numeric_cols_1_1].subtract(min_values_features_1_1, axis=0).div(ranges_features_1_1, axis=0)
# max와 min 값을 dictionary 형태로 저장
scale_min_dict_features_1_1 = min_values_features_1_1.to_dict()
scale_max_dict_features_1_1 = max_values_features_1_1.to_dict()

# 칵 column의 min 및 max 계산
min_values_features_1_2 = train_data_features_1_2[numeric_cols_1_2].min(axis=1)
max_values_features_1_2 = train_data_features_1_2[numeric_cols_1_2].max(axis=1)
# 각 행의 범위(max-min)를 계산하고, 범위가 0인 경우 1로 대체
ranges_features_1_2 = max_values_features_1_2 - min_values_features_1_2
ranges_features_1_2[ranges_features_1_2 == 0] = 1
# min-max scaling 수행
train_data_features_1_2[numeric_cols_1_2] = train_data_features_1_2[numeric_cols_1_2].subtract(min_values_features_1_2, axis=0).div(ranges_features_1_2, axis=0)
# max와 min 값을 dictionary 형태로 저장
scale_min_dict_features_1_2 = min_values_features_1_2.to_dict()
scale_max_dict_features_1_2 = max_values_features_1_2.to_dict()

# 칵 column의 min 및 max 계산
min_values_features_2_1 = train_data_features_2_1[numeric_cols_2_1].min(axis=1)
max_values_features_2_1 = train_data_features_2_1[numeric_cols_2_1].max(axis=1)
# 각 행의 범위(max-min)를 계산하고, 범위가 0인 경우 1로 대체
ranges_features_2_1 = max_values_features_2_1 - min_values_features_2_1
ranges_features_2_1[ranges_features_2_1 == 0] = 1
# min-max scaling 수행
train_data_features_2_1[numeric_cols_2_1] = train_data_features_2_1[numeric_cols_2_1].subtract(min_values_features_2_1, axis=0).div(ranges_features_2_1, axis=0)
# max와 min 값을 dictionary 형태로 저장
scale_min_dict_features_2_1 = min_values_features_2_1.to_dict()
scale_max_dict_features_2_1 = max_values_features_2_1.to_dict()

# 칵 column의 min 및 max 계산
min_values_features_2_2 = train_data_features_2_2[numeric_cols_2_2].min(axis=1)
max_values_features_2_2 = train_data_features_2_2[numeric_cols_2_2].max(axis=1)
# 각 행의 범위(max-min)를 계산하고, 범위가 0인 경우 1로 대체
ranges_features_2_2 = max_values_features_2_2 - min_values_features_2_2
ranges_features_2_2[ranges_features_2_2 == 0] = 1
# min-max scaling 수행
train_data_features_2_2[numeric_cols_2_2] = train_data_features_2_2[numeric_cols_2_2].subtract(min_values_features_2_2, axis=0).div(ranges_features_2_2, axis=0)
# max와 min 값을 dictionary 형태로 저장
scale_min_dict_features_2_2 = min_values_features_2_2.to_dict()
scale_max_dict_features_2_2 = max_values_features_2_2.to_dict()

# 칵 column의 min 및 max 계산
min_values_features_3_1 = train_data_features_3_1[numeric_cols_3_1].min(axis=1)
max_values_features_3_1 = train_data_features_3_1[numeric_cols_3_1].max(axis=1)
# 각 행의 범위(max-min)를 계산하고, 범위가 0인 경우 1로 대체
ranges_features_3_1 = max_values_features_3_1 - min_values_features_3_1
ranges_features_3_1[ranges_features_3_1 == 0] = 1
# min-max scaling 수행
train_data_features_3_1[numeric_cols_3_1] = train_data_features_3_1[numeric_cols_3_1].subtract(min_values_features_3_1, axis=0).div(ranges_features_3_1, axis=0)
# max와 min 값을 dictionary 형태로 저장
scale_min_dict_features_3_1 = min_values_features_3_1.to_dict()
scale_max_dict_features_3_1 = max_values_features_3_1.to_dict()

# 칵 column의 min 및 max 계산
min_values_features_3_2 = train_data_features_3_2[numeric_cols_3_2].min(axis=1)
max_values_features_3_2 = train_data_features_3_2[numeric_cols_3_2].max(axis=1)
# 각 행의 범위(max-min)를 계산하고, 범위가 0인 경우 1로 대체
ranges_features_3_2 = max_values_features_3_2 - min_values_features_3_2
ranges_features_3_2[ranges_features_3_2 == 0] = 1
# min-max scaling 수행
train_data_features_3_2[numeric_cols_3_2] = train_data_features_3_2[numeric_cols_3_2].subtract(min_values_features_3_2, axis=0).div(ranges_features_3_2, axis=0)
# max와 min 값을 dictionary 형태로 저장
scale_min_dict_features_3_2 = min_values_features_3_2.to_dict()
scale_max_dict_features_3_2 = max_values_features_3_2.to_dict()



label_encoder = LabelEncoder()
categorical_columns = ['대분류', '소분류', '브랜드']

for col in categorical_columns:
    label_encoder.fit(train_data[col])
    train_data[col] = label_encoder.transform(train_data[col])
label_encoder_bigcat = LabelEncoder()
train_data_2['대분류_encoded'] = label_encoder_bigcat.fit_transform(train_data_2['대분류'])

label_encoder_id = LabelEncoder()
train_data_2['소분류_encoded'] = label_encoder_id.fit_transform(train_data_2['소분류'])
train_data_2['브랜드_encoded'] = label_encoder_id.fit_transform(train_data_2['브랜드'])
train_data_2['ID_encoded'] = label_encoder_id.fit_transform(train_data_2['ID'])


def make_train_data(data,data2,data3,data4, train_size=CFG['TRAIN_WINDOW_SIZE'], predict_size=CFG['PREDICT_SIZE']):
    num_rows = len(data)
    window_size = train_size + predict_size
    input_data = np.empty((num_rows * (len(data.columns) - window_size + 1), train_size, len(data.iloc[0, :3]) + 4)) ## 여긴 분류지우는거랑 상관 없음
    target_data = np.empty((num_rows * (len(data.columns) - window_size + 1), predict_size))
    for i in tqdm(range(num_rows)):
        encode_info = np.array(data.iloc[i, :3])
        sales_data = np.array(data.iloc[i, 3:])
        feature1_data = np.array(data2.iloc[i, 3:])
        feature2_data = np.array(data3.iloc[i, 3:])
        feature3_data = np.array(data4.iloc[i, 3:])

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
    input_data = np.empty((num_rows, train_size, len(data.iloc[0, :3]) + 4))
    for i in tqdm(range(num_rows)):
        encode_info = np.array(data.iloc[i, :3])
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

def create_new_val_loader_modified(data,data2,data3,data4, train_size=CFG['TRAIN_WINDOW_SIZE'], predict_size=CFG['PREDICT_SIZE'], feature_size=7):
    num_rows = len(data)
    input_data = np.empty((num_rows, train_size, feature_size))
    target_data = np.empty((num_rows, predict_size))
    for i, (_, row) in enumerate(data.iterrows()):
        encode_info = np.array([row['대분류_encoded'], row['소분류_encoded'], row['브랜드_encoded']])
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



train_input, train_target = make_train_data(train_data,train_data_features_1_1,train_data_features_2_1,train_data_features_3_1)
test_input = make_predict_data(train_data,train_data_features_1_1,train_data_features_2_1,train_data_features_3_1)

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
new_val_loader = create_new_val_loader_modified(train_data_2,train_data_features_1_2,train_data_features_2_2,train_data_features_3_2)

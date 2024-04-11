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



### 1개

def make_train_data(data, train_size=CFG['TRAIN_WINDOW_SIZE'], predict_size=CFG['PREDICT_SIZE'], feature_size = 3):
    num_rows = len(data)
    window_size = train_size + predict_size
    series_size = len(data.iloc[0, feature_size:]) - window_size + 1
    input_data = np.empty((num_rows * series_size, train_size, len(data.iloc[0, :feature_size]) + 1))
    target_data = np.empty((num_rows * series_size, predict_size))
    for i in tqdm(range(num_rows)):
        encode_info = np.array(data.iloc[i, :feature_size])
        sales_data = np.array(data.iloc[i, feature_size:])
        for j in range(len(sales_data) - window_size + 1):
            window = sales_data[j : j + window_size]
            temp_data = np.column_stack((np.tile(encode_info, (train_size, 1)), window[:train_size]))
            input_data[i * series_size + j] = temp_data
            target_data[i * series_size + j] = window[train_size:]
    return input_data, target_data, series_size

def make_predict_data(data, train_size=CFG['TRAIN_WINDOW_SIZE'], feature_size=3):
    num_rows = len(data)
    input_data = np.empty((num_rows, train_size, len(data.iloc[0, :feature_size]) + 1))
    for i in tqdm(range(num_rows)):
        encode_info = np.array(data.iloc[i, :feature_size])
        sales_data = np.array(data.iloc[i, -train_size:])
        window = sales_data[-train_size : ]
        temp_data = np.column_stack((np.tile(encode_info, (train_size, 1)), window[:train_size]))
        input_data[i] = temp_data
    return input_data

def create_new_val_loader_modified(data, train_size=CFG['TRAIN_WINDOW_SIZE'], predict_size=CFG['PREDICT_SIZE'], feature_size=4):
    num_rows = len(data)
    input_data = np.empty((num_rows, train_size, feature_size))
    target_data = np.empty((num_rows, predict_size))
    for i, (_, row) in enumerate(data.iterrows()):
        encode_info = np.array([row['대분류_encoded'], row['소분류_encoded'], row['브랜드_encoded']])
        sales_data = np.array(row[-(train_size + predict_size):])
        if len(sales_data) < train_size + predict_size:
            print(f"Row {i} has insufficient data. Skipping...")
            continue
        window = sales_data[:train_size]
        temp_data = np.column_stack((np.tile(encode_info, (train_size, 1)), window))
        input_data[i] = temp_data
        target_data[i] = sales_data[train_size:]
    dataset = CustomDataset_2(input_data, target_data)
    loader = DataLoader(dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
    return loader


import pandas as pd
from tqdm import tqdm

specific_list_a = [
    "B002-C003-0007",
    "B002-C003-0009",
    "B002-C003-0010",
    "B002-C003-0011",
    "B002-C003-0012",
    "B002-C003-0016",
    "B002-C003-0017"
]

specific_list_b = [
    "B002-C003-0008",
    "B002-C003-0013",
    "B002-C003-0014",
    "B002-C003-0015",
    "B002-C003-0018",
    "B002-C003-0048"
]

specific_list_c = [
    "B002-C003-0019",
    "B002-C003-0020",
    "B002-C003-0021",
    "B002-C003-0022",
    "B002-C003-0023"
]

specific_list_d = [
    "B002-C003-0024",
    "B002-C003-0026",
    "B002-C003-0029"
]

specific_list_e = [
    "B002-C003-0030",
    "B002-C003-0031",
    "B002-C003-0033"
]

specific_list_f = [
    "B002-C003-0035",
    "B002-C003-0036",
    "B002-C003-0037",
    "B002-C003-0038",
    "B002-C003-0039",
    "B002-C003-0040",
]

specific_list_g = [
    "B002-C003-0034",
    "B002-C003-0045",
    "B002-C003-0050",
]

specific_list_h = [
    "B002-C003-0001",
    "B002-C003-0002",
    "B002-C003-0003",
    "B002-C003-0004",
    "B002-C003-0005",
    "B002-C003-0006"
]

specific_list_i = [
    "B002-C003-0025",
    "B002-C003-0027",
    "B002-C003-0028",
    "B002-C003-0032",
    "B002-C003-0041",
    "B002-C003-0042",
    "B002-C003-0043",
    "B002-C003-0044",
    "B002-C003-0046",
    "B002-C003-0046",  # This value is repeated, consider removing if not intentional
    "B002-C003-0047",
    "B002-C003-0049",
    "B002-C003-0051",
    "B002-C003-0052",
    "B002-C003-0053"
]





train_sample = pd.read_csv('./10.omega/input/train.csv')


specific_list_i = [
    "B002-C003-0025",
    "B002-C003-0027",
    "B002-C003-0028",
    "B002-C003-0032",
    "B002-C003-0041",
    "B002-C003-0042",
    "B002-C003-0043",
    "B002-C003-0044",
    "B002-C003-0046",
    "B002-C003-0046",  # This value is repeated, consider removing if not intentional
    "B002-C003-0047",
    "B002-C003-0049",
    "B002-C003-0051",
    "B002-C003-0052",
    "B002-C003-0053"
]


train_sample = pd.read_csv('./10.omega/input/train.csv')

# 2. Filter the dataframe
filtered_df = train_sample[train_sample["소분류"].isin(specific_list_a)]

# Save to CSV
filtered_df.to_csv('./10.omega/6.sobun/train_a.csv', index=False)

submission_sample = pd.read_csv('./10.omega/6.sobun/sample_submission.csv')
filtered_ids = filtered_df['ID'].values

# Filter the submission_sample DataFrame based on the IDs
submission = submission_sample[submission_sample['ID'].isin(filtered_ids)]

# Save the filtered submission to CSV
submission_path = './10.omega/temp_output/sample_submission_a.csv'
submission.to_csv(submission_path, index=False)

brand_sample = pd.read_csv('./10.omega/temp_output/sales_price_real.csv')

brand_dict = {}
for index, row in tqdm(brand_sample.iterrows()):
    brand_name = row[0]
    brand_data = row[1:].values
    brand_dict[brand_name] = brand_data

for index, row in tqdm(filtered_df.iterrows()):
    brand_name = row[5]  # 브랜드 column
    if brand_name in brand_dict:
        if len(brand_dict[brand_name]) != len(filtered_df.columns) - 7:
            print(f"Skipping {brand_name} due to size mismatch.")
            continue
        filtered_df.loc[index, filtered_df.columns[7:]] = brand_dict[brand_name]


filtered_df.fillna(0, inplace=True) ## brand data가 없는 경우가 있어서 nan을 0으로 처리합니다 
brand_path = './10.omega/temp_output/sales_a.csv'
filtered_df.to_csv(brand_path, index=False)

## brand data 결과
brand_path




train_sample = pd.read_csv('./10.omega/input/train.csv')

# 2. Filter the dataframe
filtered_df = train_sample[train_sample["소분류"].isin(specific_list_b)]

# Save to CSV
filtered_df.to_csv('./10.omega/6.sobun/train_b.csv', index=False)

submission_sample = pd.read_csv('./10.omega/6.sobun/sample_submission.csv')
filtered_ids = filtered_df['ID'].values

# Filter the submission_sample DataFrame based on the IDs
submission = submission_sample[submission_sample['ID'].isin(filtered_ids)]

# Save the filtered submission to CSV
submission_path = './10.omega/temp_output/sample_submission_b.csv'
submission.to_csv(submission_path, index=False)

brand_sample = pd.read_csv('./10.omega/temp_output/sales_price_real.csv')

brand_dict = {}
for index, row in tqdm(brand_sample.iterrows()):
    brand_name = row[0]
    brand_data = row[1:].values
    brand_dict[brand_name] = brand_data

for index, row in tqdm(filtered_df.iterrows()):
    brand_name = row[5]  # 브랜드 column
    if brand_name in brand_dict:
        if len(brand_dict[brand_name]) != len(filtered_df.columns) - 7:
            print(f"Skipping {brand_name} due to size mismatch.")
            continue
        filtered_df.loc[index, filtered_df.columns[7:]] = brand_dict[brand_name]


filtered_df.fillna(0, inplace=True) ## brand data가 없는 경우가 있어서 nan을 0으로 처리합니다 
brand_path = './10.omega/temp_output/sales_b.csv'
filtered_df.to_csv(brand_path, index=False)

## brand data 결과
brand_path





train_sample = pd.read_csv('./10.omega/input/train.csv')

# 2. Filter the dataframe
filtered_df = train_sample[train_sample["소분류"].isin(specific_list_c)]

# Save to CSV
filtered_df.to_csv('./10.omega/6.sobun/train_c.csv', index=False)

submission_sample = pd.read_csv('./10.omega/6.sobun/sample_submission.csv')
filtered_ids = filtered_df['ID'].values

# Filter the submission_sample DataFrame based on the IDs
submission = submission_sample[submission_sample['ID'].isin(filtered_ids)]

# Save the filtered submission to CSV
submission_path = './10.omega/temp_output/sample_submission_c.csv'
submission.to_csv(submission_path, index=False)

brand_sample = pd.read_csv('./10.omega/temp_output/sales_price_real.csv')

brand_dict = {}
for index, row in tqdm(brand_sample.iterrows()):
    brand_name = row[0]
    brand_data = row[1:].values
    brand_dict[brand_name] = brand_data

for index, row in tqdm(filtered_df.iterrows()):
    brand_name = row[5]  # 브랜드 column
    if brand_name in brand_dict:
        if len(brand_dict[brand_name]) != len(filtered_df.columns) - 7:
            print(f"Skipping {brand_name} due to size mismatch.")
            continue
        filtered_df.loc[index, filtered_df.columns[7:]] = brand_dict[brand_name]


filtered_df.fillna(0, inplace=True) ## brand data가 없는 경우가 있어서 nan을 0으로 처리합니다 
brand_path = './10.omega/temp_output/sales_c.csv'
filtered_df.to_csv(brand_path, index=False)

## brand data 결과
brand_path




train_sample = pd.read_csv('./10.omega/input/train.csv')

# 2. Filter the dataframe
filtered_df = train_sample[train_sample["소분류"].isin(specific_list_d)]

# Save to CSV
filtered_df.to_csv('./10.omega/6.sobun/train_d.csv', index=False)

submission_sample = pd.read_csv('./10.omega/6.sobun/sample_submission.csv')
filtered_ids = filtered_df['ID'].values

# Filter the submission_sample DataFrame based on the IDs
submission = submission_sample[submission_sample['ID'].isin(filtered_ids)]

# Save the filtered submission to CSV
submission_path = './10.omega/temp_output/sample_submission_d.csv'
submission.to_csv(submission_path, index=False)

brand_sample = pd.read_csv('./10.omega/temp_output/sales_price_real.csv')

brand_dict = {}
for index, row in tqdm(brand_sample.iterrows()):
    brand_name = row[0]
    brand_data = row[1:].values
    brand_dict[brand_name] = brand_data

for index, row in tqdm(filtered_df.iterrows()):
    brand_name = row[5]  # 브랜드 column
    if brand_name in brand_dict:
        if len(brand_dict[brand_name]) != len(filtered_df.columns) - 7:
            print(f"Skipping {brand_name} due to size mismatch.")
            continue
        filtered_df.loc[index, filtered_df.columns[7:]] = brand_dict[brand_name]


filtered_df.fillna(0, inplace=True) ## brand data가 없는 경우가 있어서 nan을 0으로 처리합니다 
brand_path = './10.omega/temp_output/sales_d.csv'
filtered_df.to_csv(brand_path, index=False)

## brand data 결과
brand_path



train_sample = pd.read_csv('./10.omega/input/train.csv')

# 2. Filter the dataframe
filtered_df = train_sample[train_sample["소분류"].isin(specific_list_e)]

# Save to CSV
filtered_df.to_csv('./10.omega/6.sobun/train_e.csv', index=False)

submission_sample = pd.read_csv('./10.omega/6.sobun/sample_submission.csv')
filtered_ids = filtered_df['ID'].values

# Filter the submission_sample DataFrame based on the IDs
submission = submission_sample[submission_sample['ID'].isin(filtered_ids)]

# Save the filtered submission to CSV
submission_path = './10.omega/temp_output/sample_submission_e.csv'
submission.to_csv(submission_path, index=False)

brand_sample = pd.read_csv('./10.omega/temp_output/sales_price_real.csv')

brand_dict = {}
for index, row in tqdm(brand_sample.iterrows()):
    brand_name = row[0]
    brand_data = row[1:].values
    brand_dict[brand_name] = brand_data

for index, row in tqdm(filtered_df.iterrows()):
    brand_name = row[5]  # 브랜드 column
    if brand_name in brand_dict:
        if len(brand_dict[brand_name]) != len(filtered_df.columns) - 7:
            print(f"Skipping {brand_name} due to size mismatch.")
            continue
        filtered_df.loc[index, filtered_df.columns[7:]] = brand_dict[brand_name]


filtered_df.fillna(0, inplace=True) ## brand data가 없는 경우가 있어서 nan을 0으로 처리합니다 
brand_path = './10.omega/temp_output/sales_e.csv'
filtered_df.to_csv(brand_path, index=False)

## brand data 결과
brand_path




train_sample = pd.read_csv('./10.omega/input/train.csv')

# 2. Filter the dataframe
filtered_df = train_sample[train_sample["소분류"].isin(specific_list_f)]

# Save to CSV
filtered_df.to_csv('./10.omega/6.sobun/train_f.csv', index=False)

submission_sample = pd.read_csv('./10.omega/6.sobun/sample_submission.csv')
filtered_ids = filtered_df['ID'].values

# Filter the submission_sample DataFrame based on the IDs
submission = submission_sample[submission_sample['ID'].isin(filtered_ids)]

# Save the filtered submission to CSV
submission_path = './10.omega/temp_output/sample_submission_f.csv'
submission.to_csv(submission_path, index=False)

brand_sample = pd.read_csv('./10.omega/temp_output/sales_price_real.csv')

brand_dict = {}
for index, row in tqdm(brand_sample.iterrows()):
    brand_name = row[0]
    brand_data = row[1:].values
    brand_dict[brand_name] = brand_data

for index, row in tqdm(filtered_df.iterrows()):
    brand_name = row[5]  # 브랜드 column
    if brand_name in brand_dict:
        if len(brand_dict[brand_name]) != len(filtered_df.columns) - 7:
            print(f"Skipping {brand_name} due to size mismatch.")
            continue
        filtered_df.loc[index, filtered_df.columns[7:]] = brand_dict[brand_name]


filtered_df.fillna(0, inplace=True) ## brand data가 없는 경우가 있어서 nan을 0으로 처리합니다 
brand_path = './10.omega/temp_output/sales_f.csv'
filtered_df.to_csv(brand_path, index=False)

## brand data 결과
brand_path




train_sample = pd.read_csv('./10.omega/input/train.csv')

# 2. Filter the dataframe
filtered_df = train_sample[train_sample["소분류"].isin(specific_list_g)]

# Save to CSV
filtered_df.to_csv('./10.omega/6.sobun/train_g.csv', index=False)

submission_sample = pd.read_csv('./10.omega/6.sobun/sample_submission.csv')
filtered_ids = filtered_df['ID'].values

# Filter the submission_sample DataFrame based on the IDs
submission = submission_sample[submission_sample['ID'].isin(filtered_ids)]

# Save the filtered submission to CSV
submission_path = './10.omega/temp_output/sample_submission_g.csv'
submission.to_csv(submission_path, index=False)

brand_sample = pd.read_csv('./10.omega/temp_output/sales_price_real.csv')

brand_dict = {}
for index, row in tqdm(brand_sample.iterrows()):
    brand_name = row[0]
    brand_data = row[1:].values
    brand_dict[brand_name] = brand_data

for index, row in tqdm(filtered_df.iterrows()):
    brand_name = row[5]  # 브랜드 column
    if brand_name in brand_dict:
        if len(brand_dict[brand_name]) != len(filtered_df.columns) - 7:
            print(f"Skipping {brand_name} due to size mismatch.")
            continue
        filtered_df.loc[index, filtered_df.columns[7:]] = brand_dict[brand_name]


filtered_df.fillna(0, inplace=True) ## brand data가 없는 경우가 있어서 nan을 0으로 처리합니다 
brand_path = './10.omega/temp_output/sales_g.csv'
filtered_df.to_csv(brand_path, index=False)

## brand data 결과
brand_path





train_sample = pd.read_csv('./10.omega/input/train.csv')

# 2. Filter the dataframe
filtered_df = train_sample[train_sample["소분류"].isin(specific_list_h)]

# Save to CSV
filtered_df.to_csv('./10.omega/6.sobun/train_h.csv', index=False)

submission_sample = pd.read_csv('./10.omega/6.sobun/sample_submission.csv')
filtered_ids = filtered_df['ID'].values

# Filter the submission_sample DataFrame based on the IDs
submission = submission_sample[submission_sample['ID'].isin(filtered_ids)]

# Save the filtered submission to CSV
submission_path = './10.omega/temp_output/sample_submission_h.csv'
submission.to_csv(submission_path, index=False)

brand_sample = pd.read_csv('./10.omega/temp_output/sales_price_real.csv')

brand_dict = {}
for index, row in tqdm(brand_sample.iterrows()):
    brand_name = row[0]
    brand_data = row[1:].values
    brand_dict[brand_name] = brand_data

for index, row in tqdm(filtered_df.iterrows()):
    brand_name = row[5]  # 브랜드 column
    if brand_name in brand_dict:
        if len(brand_dict[brand_name]) != len(filtered_df.columns) - 7:
            print(f"Skipping {brand_name} due to size mismatch.")
            continue
        filtered_df.loc[index, filtered_df.columns[7:]] = brand_dict[brand_name]


filtered_df.fillna(0, inplace=True) ## brand data가 없는 경우가 있어서 nan을 0으로 처리합니다 
brand_path = './10.omega/temp_output/sales_h.csv'
filtered_df.to_csv(brand_path, index=False)

## brand data 결과
brand_path





import pandas as pd
from tqdm import tqdm
train_sample = pd.read_csv('./10.omega/input/train.csv')


specific_list_i = [
    "B002-C003-0025",
    "B002-C003-0027",
    "B002-C003-0028",
    "B002-C003-0032",
    "B002-C003-0041",
    "B002-C003-0042",
    "B002-C003-0043",
    "B002-C003-0044",
    "B002-C003-0046",
    "B002-C003-0046",  # This value is repeated, consider removing if not intentional
    "B002-C003-0047",
    "B002-C003-0049",
    "B002-C003-0051",
    "B002-C003-0052",
    "B002-C003-0053"
]



# 2. Filter the dataframe
filtered_df = train_sample[train_sample["소분류"].isin(specific_list_i)]

# Save to CSV
filtered_df.to_csv('./10.omega/6.sobun/train_i.csv', index=False)

submission_sample = pd.read_csv('./10.omega/6.sobun/sample_submission.csv')
filtered_ids = filtered_df['ID'].values

# Filter the submission_sample DataFrame based on the IDs
submission = submission_sample[submission_sample['ID'].isin(filtered_ids)]

# Save the filtered submission to CSV
submission_path = './10.omega/temp_output/sample_submission_i.csv'
submission.to_csv(submission_path, index=False)

brand_sample = pd.read_csv('./10.omega/temp_output/sales_price_real.csv')

brand_dict = {}
for index, row in tqdm(brand_sample.iterrows()):
    brand_name = row[0]
    brand_data = row[1:].values
    brand_dict[brand_name] = brand_data

for index, row in tqdm(filtered_df.iterrows()):
    brand_name = row[5]  # 브랜드 column
    if brand_name in brand_dict:
        if len(brand_dict[brand_name]) != len(filtered_df.columns) - 7:
            print(f"Skipping {brand_name} due to size mismatch.")
            continue
        filtered_df.loc[index, filtered_df.columns[7:]] = brand_dict[brand_name]


filtered_df.fillna(0, inplace=True) ## brand data가 없는 경우가 있어서 nan을 0으로 처리합니다 
brand_path = './10.omega/temp_output/sales_i.csv'
filtered_df.to_csv(brand_path, index=False)

## brand data 결과
brand_path


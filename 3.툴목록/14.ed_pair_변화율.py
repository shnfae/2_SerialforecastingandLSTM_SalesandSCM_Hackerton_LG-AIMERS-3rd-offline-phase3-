import pandas as pd
import numpy as np
from tqdm import tqdm


train_df = pd.read_csv("./codereview/train_sample.csv")

try:
    sales_df = pd.read_csv("./codereview/sales_sample.csv", encoding='ISO-8859-1')
except:
    sales_df = pd.read_csv("./codereview/sales_sample.csv", encoding='cp1252')


sales_df.iloc[:, 1:] = sales_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
sales_df.iloc[:, 1:] = np.ceil(sales_df.iloc[:, 1:] / 100) * 100

train_columns = train_df.columns
sales_columns = sales_df.columns


# Step 1: Filter out zero sales or quantities and create pairs
# Convert long format data to wide format for easier processing
train_long = train_df.melt(id_vars=['ID'], value_vars=train_columns[1:], var_name='d', value_name='train')
sales_long = sales_df.melt(id_vars=['ID'], value_vars=sales_columns[1:], var_name='d', value_name='sales')

# Merge the reshaped dataframes

# Merge the reshaped dataframes
merged_df = train_long.merge(sales_long, on=['ID', 'd'], how='left')

# Ensure 'train' and 'sales' columns are numeric
merged_df['train'] = pd.to_numeric(merged_df['train'], errors='coerce')
merged_df['sales'] = pd.to_numeric(merged_df['sales'], errors='coerce')

# Convert NaN values to 0
merged_df.fillna(0, inplace=True)

# 2. Create pairs, filtering out entries where either quantity or sale is zero
filtered_df = merged_df[(merged_df['train'] != 0) & (merged_df['sales'] != 0)]
pairs_dict = {}

for idx, group in tqdm(filtered_df.groupby('ID')):
    pairs = list(zip(group['train'], group['sales']))
    pairs_dict[idx] = pairs

# 3. Sort the pairs for each ID
for idx in tqdm(pairs_dict):
    pairs_dict[idx].sort(key=lambda x: (x[0], -x[1]))  # Sort by quantity asc and sales desc

print(pairs_dict[0])
# 4. Calculate elasticity for each pair
new_pairs_dict = {}

for idx, pairs in tqdm(pairs_dict.items()):
    aggregation = {}
    counts = {}  # This dictionary will keep track of the counts of each sale value
    for q, s in tqdm(pairs):
        if s in aggregation:
            aggregation[s] += q
            counts[s] += 1
        else:
            aggregation[s] = q
            counts[s] = 1

    # Convert the aggregated sum to average
    for s in tqdm(aggregation):
        aggregation[s] = round(aggregation[s] / counts[s])
            
    # Convert the aggregated dictionary back to list of pairs
    aggregated_pairs = [(q, s) for s, q in aggregation.items()]
    aggregated_pairs.sort(key=lambda x: (-x[1], x[0]))  # Sort by sales desc and then by quantity
    new_pairs_dict[idx] = aggregated_pairs

# Calculate elasticity for each aggregated pair
print(new_pairs_dict[99])

new_elasticity_dict = {}

for idx, pairs in tqdm(new_pairs_dict.items()):
    elasticity_list = []
    for i in range(len(pairs) - 1):
        q1, p1 = pairs[i]
        q2, p2 = pairs[i+1]
        if p2 - p1 == 0:  # Avoid division by zero
            elasticity = np.nan
        else:
            price_elasticity = (p2 - p1) / p1
            quantity_elasticity = (q2 - q1) / q1
            elasticity = quantity_elasticity / price_elasticity
        elasticity_list.append(elasticity)
    new_elasticity_dict[idx] = elasticity_list

all_ids = set(train_df['ID'].unique())
missing_ids = all_ids - set(new_elasticity_dict.keys())

# Add missing IDs with NaN values
for idx in missing_ids:
    new_elasticity_dict[idx] = [np.nan]


new_elasticities_all = {idx: elasticity_list for idx, elasticity_list in new_elasticity_dict.items()}

# Compute average elasticity for each ID
new_avg_elasticity = {idx: np.nanmean(elasticities) if len(elasticities) != 0 else np.nan for idx, elasticities in new_elasticity_dict.items()}
new_avg_elasticity_df = pd.DataFrame(list(new_avg_elasticity.items()), columns=['ID', 'Average_Elasticity'])


# Save results to CSV
elasticity_df = pd.DataFrame.from_dict(new_elasticities_all, orient='index').transpose()


elasticity_path = './codereview/ed_pair_full_revised_4.csv'
avg_elasticity_path = './codereview/ed_pair_avg_revised_4.csv'

elasticity_df.to_csv(elasticity_path, index=False)
new_avg_elasticity_df.to_csv(avg_elasticity_path, index=False)

elasticity_path, avg_elasticity_path

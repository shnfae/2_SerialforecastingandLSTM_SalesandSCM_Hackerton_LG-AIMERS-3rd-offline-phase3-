import pandas as pd
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm


train_df = pd.read_csv("./10.omega/input/train_rev.csv")

try:
    sales_df = pd.read_csv("./10.omega/temp_output/sales_price.csv", encoding='ISO-8859-1')
except:
    sales_df = pd.read_csv("./10.omega/temp_output/sales_price.csv", encoding='cp1252')




sales_df.iloc[:, 1:] = sales_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')



def custom_rounding(x):
    if 100 <= x < 1000:  # 4-digit number
        return np.round(x / 10) * 10
    elif 1000 <= x < 10000:
        return np.round(x / 100) * 100
    elif 10000 <= x < 100000:  # 5-digit number
        return np.round(x / 1000) * 1000
    elif 100000 <= x < 1000000:  # 5-digit number
        return np.round(x / 10000) * 10000
    elif 1000000 <= x < 10000000:  # 5-digit number
        return np.round(x / 100000) * 100000
    elif 10000000 <= x < 100000000:  # 5-digit number
        return np.round(x / 1000000) * 1000000

# Apply custom rounding to all prices in the sales dataframe
sales_df.iloc[:, 1:] = sales_df.iloc[:, 1:].applymap(custom_rounding)


train_columns = train_df.columns
sales_columns = sales_df.columns

sales_df.rename(columns={'ï»¿ID': 'ID'}, inplace=True)


# Step 1: Filter out zero sales or quantities and create pairs
# Convert long format data to wide format for easier processing
train_long = train_df.melt(id_vars=['ID'], value_vars=train_columns[1:], var_name='d', value_name='train')
sales_long = sales_df.melt(id_vars=['ID'], value_vars=sales_columns[1:], var_name='d', value_name='sales')

# Merge the reshaped dataframes

# Merge the reshaped dataframes
# ... [previous code]

# Merge the reshaped dataframes
merged_df = train_long.merge(sales_long, on=['ID', 'd'], how='left')

print("Merged DF shape:", merged_df.shape)
print(merged_df.head())

# Ensure 'train' and 'sales' columns are numeric
merged_df['train'] = pd.to_numeric(merged_df['train'], errors='coerce')
merged_df['sales'] = pd.to_numeric(merged_df['sales'], errors='coerce')

# Convert NaN values to 0
merged_df.fillna(0, inplace=True)

# 2. Create pairs, filtering out entries where either quantity or sale is zero
filtered_df = merged_df[(merged_df['train'] != 0) & (merged_df['sales'] != 0)]

print("Filtered DF shape:", filtered_df.shape)
print(filtered_df.head())

pairs_dict = {}
for idx, group in tqdm(filtered_df.groupby('ID')):
    pairs = list(zip(group['train'], group['sales']))
    pairs_dict[idx] = pairs

print("Keys in pairs_dict:", list(pairs_dict.keys()))

# Accessing the first item in pairs_dict
first_key = list(pairs_dict.keys())[0]
print(pairs_dict[first_key])

# 3. Sort the pairs for each ID
for idx in tqdm(pairs_dict):
    pairs_dict[idx].sort(key=lambda x: (x[0], -x[1]))  # Sort by quantity asc and sales desc

# ... [rest of the code]

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
    df_pairs = pd.DataFrame(pairs, columns=['quantity', 'price'])
    df_pairs = df_pairs.dropna()  # Drop rows with NaN values
    
    # Take the log of quantity and price
    log_quantity = np.log(df_pairs['quantity'])
    log_price = sm.add_constant(np.log(df_pairs['price']))  # Adds a constant to the predictor
    
    # Skip if dataframe has too few rows
    if len(df_pairs) < 2:
        new_elasticity_dict[idx] = [np.nan]
        continue
    
    # Fit the regression model
    model = sm.OLS(log_quantity, log_price).fit()
    
    # Extract the elasticity (coefficient of log_price)
    elasticity = model.params[1]
    new_elasticity_dict[idx] = [elasticity]

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


elasticity_path = './10.omega/ed_pair_full_reg_omega.csv'
avg_elasticity_path = './10.omega/ed_pair_full_reg_omega_avg.csv'

elasticity_df.to_csv(elasticity_path, index=False)
new_avg_elasticity_df.to_csv(avg_elasticity_path, index=False)

elasticity_path, avg_elasticity_path

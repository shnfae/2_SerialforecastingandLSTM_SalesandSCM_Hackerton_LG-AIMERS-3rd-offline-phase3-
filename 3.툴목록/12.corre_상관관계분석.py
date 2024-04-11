import pandas as pd
import numpy as np
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load data
train_sample = pd.read_csv("./codereview/train.csv")
brand_sample = pd.read_csv("./codereview/brand.csv")
sales_sample = pd.read_csv("./codereview/sales_price.csv")
weekday_sample = pd.read_csv("./codereview/weekday.csv")

quantities = train_sample.iloc[:, 6:].values.flatten()
brand_points = brand_sample.iloc[:, 6:].values.flatten()
sales = sales_sample.iloc[:, 6:].values.flatten()
weekday = weekday_sample.iloc[:, 6:].values.flatten()

# Tile the categorical columns to match the length of the flattened arrays
num_repeats = train_sample.iloc[:, 6:].shape[1]
대분류 = np.tile(train_sample['대분류'].values, num_repeats)
중분류 = np.tile(train_sample['중분류'].values, num_repeats)
소분류 = np.tile(train_sample['소분류'].values, num_repeats)
브랜드 = np.tile(train_sample['브랜드'].values, num_repeats)

print("quantities:", quantities.shape)
print("brand_points:", brand_points.shape)
print("sales:", sales.shape)
print("weekday:", weekday.shape)
print("대분류:", 대분류.shape)
print("중분류:", 중분류.shape)
print("소분류:", 소분류.shape)
print("브랜드:", 브랜드.shape)

# Combine into a DataFrame
df = pd.DataFrame({
    'quantities': quantities,
    'brand_points': brand_points,
    'sales': sales,
    'weekday': weekday,
    'big': 대분류,
    'mid': 중분류,
    'sma': 소분류,
    'brand': 브랜드
})
# Compute correlation
corr_matrix = df.corr()


# Cramer's V for categorical columns
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

categorical_features = ['big', 'mid', 'sma', 'brand']
categorical_corr = pd.DataFrame(index=categorical_features, columns=categorical_features)
for col1 in categorical_features:
    for col2 in categorical_features:
        categorical_corr.loc[col1, col2] = cramers_v(df[col1], df[col2])

# Merge numerical and categorical correlations
final_corr = corr_matrix.join(categorical_corr, how='outer').fillna(1)

# Display the correlation matrix
plt.figure(figsize=(12,10))
sns.heatmap(final_corr, annot=True, cmap='coolwarm')
plt.show()

all_correlations = []
correlation_matrix=[]

all_correlation_matrices = []

for idx in tqdm(range(len(train_sample))):
    quantities_for_id = pd.to_numeric(train_sample.iloc[idx, 6:], errors='coerce').values
    sales_for_id = pd.to_numeric(sales_sample.iloc[idx, 6:], errors='coerce').values
    brand_for_id = pd.to_numeric(brand_sample.iloc[idx, 6:], errors='coerce').values
    weekday_for_id = pd.to_numeric(weekday_sample.iloc[idx, 6:], errors='coerce').values

    # Stack the series to make a 2D array
    data_for_id = np.vstack([quantities_for_id, sales_for_id, brand_for_id, weekday_for_id])
    
    # Check for NaN values
    if np.isnan(data_for_id).any():
        print(f"Warning: Data for idx {idx} contains NaN values. Skipping...")
        continue
    
    try:
        correlation_matrix = np.corrcoef(data_for_id)
        all_correlation_matrices.append(correlation_matrix)
    except Exception as e:
        print(f"Error encountered for idx {idx}: {e}")

all_correlation_matrices_array = np.array(all_correlation_matrices)

# Compute the mean correlation matrix while ignoring NaN values
average_correlation_matrix = np.nanmean(all_correlation_matrices_array, axis=0)

# Check and handle NaN values in the average_correlation_matrix
if np.isnan(average_correlation_matrix).any():
    print("Warning: Average correlation matrix contains NaN values!")
    average_correlation_matrix = np.nan_to_num(average_correlation_matrix)  # Replace NaN values with 0

print(average_correlation_matrix)

labels = ['quantities', 'sales', 'brand_points', 'weekday']
plt.figure(figsize=(12, 10))
sns.heatmap(average_correlation_matrix, annot=True, cmap='coolwarm', xticklabels=labels, yticklabels=labels)
plt.show()

all_matrices_df = pd.DataFrame()

for idx, matrix in enumerate(all_correlation_matrices):
    df = pd.DataFrame(matrix, columns=labels, index=labels)
    df = df.melt(ignore_index=False).reset_index()
    df['ID'] = idx
    all_matrices_df = all_matrices_df.append(df)

all_matrices_df.to_csv("./codereview/all_correlation_matrices.csv", index=False)

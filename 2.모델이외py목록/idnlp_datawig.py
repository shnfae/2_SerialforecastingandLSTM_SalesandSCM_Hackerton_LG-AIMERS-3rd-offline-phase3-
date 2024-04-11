import pandas as pd
import datawig

# Load the data and fill any NaN values with empty strings
df = pd.read_csv("/data/user/codereview/nlp/feature_aug_all_filled_revised.csv", encoding='utf-8')
df = df.fillna('').astype(str)

# Define the columns for imputation
all_columns = df.columns.tolist() 
output_column = 'human' 
input_columns = [col for col in all_columns if col != output_column]

# Initialize and train the imputer
imputer = datawig.SimpleImputer(
    input_columns=input_columns,  # columns used for imputation
    output_column=output_column   # column with missing values to be imputed
)
imputer.fit(train_df=df, num_epochs=50)

# Extract rows where the output_column has NaN values
df = df.reset_index(drop=True)
df_null_only = df[df[output_column] == '']
df_null_only.reset_index(drop=True, inplace=True)

df = df.astype(str)
df_null_only = df_null_only.astype(str)

print("NaN values in df:", df.isnull().sum())
print("NaN values in df_null_only:", df_null_only.isnull().sum())

# Predict the missing values
np_imputed = imputer.predict(df_null_only)
df_imputed = pd.DataFrame(np_imputed)

# Merge the imputed values with the original dataframe
df_merged = df.merge(df_imputed[['id', 'human_imputed', 'human_imputed_proba']], on='id', how='left')
df_merged.drop(columns=['id'], inplace=True)

# Check the imputed data
print(df_imputed.head(10))

# Save the imputed data
df_imputed.to_csv("/data/user/codereview/nlp/feature_aug_after_human_revised_1_realfinal.csv", encoding='utf-8-sig', index_label="id")

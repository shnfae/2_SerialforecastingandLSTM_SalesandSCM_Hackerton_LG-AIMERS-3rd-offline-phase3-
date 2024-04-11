import pandas as pd

# Load the data from a CSV file
train_df = pd.read_csv("./codereview/train_sample.csv")

sum_list = []
idx_list_temp=[]

for _, row in train_df.iterrows():
    idx_list = [row['대분류'], row['중분류'], row['소분류']]
    if idx_list not in sum_list:
        sum_list.append(idx_list)
    idx_list_temp.append(idx_list)

print(sum_list)

print(len(sum_list))

print(idx_list_temp[4],idx_list_temp[9561])

new_type = []
for _, row in train_df.iterrows():
    idx_list = [row['대분류'], row['중분류'], row['소분류']]
    k = sum_list.index(idx_list)
    new_type.append([row['ID'], k])

new_df = pd.DataFrame(new_type, columns=['ID', 'new_type'])
new_df.to_csv("./codereview/id_newtype.csv", index=False)

print(new_type[4])
print(new_type[9561])
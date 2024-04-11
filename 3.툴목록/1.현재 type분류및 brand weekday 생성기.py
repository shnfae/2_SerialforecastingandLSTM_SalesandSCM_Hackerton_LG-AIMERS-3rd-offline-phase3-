import numpy as np
import pandas as pd
from tqdm.auto import tqdm

#2. type 2, 3 분리하기
## 조금 복잡하지만 잘 설명해보겠습니다

## 1) 파일을 불러오고 관련 배열작업을 해줍니다.
train_sample = pd.read_csv('./10.omega/input/train.csv')
try:
    sales_sample = pd.read_csv('./10.omega/input/sales.csv', encoding='cp949')
except:
    sales_sample = pd.read_csv('./10.omega/input/sales.csv', encoding='ISO-8859-1')
np.seterr(divide='ignore', invalid='ignore')
train_sample.iloc[:, 7:] = train_sample.iloc[:, 7:].apply(pd.to_numeric, errors='coerce')
sales_sample.iloc[:, 7:] = sales_sample.iloc[:, 7:].apply(pd.to_numeric, errors='coerce')

## 2) 일자별 가격을 구합니다.
### sales에 train을 나눕니다. 즉 sales는 매출(가격 x 수량) 이기에 train(수량)을 나눠주면 각 일자별, id별 가격이나옵니다
### 단, train이 0일수 있으므로 (판매량이 0일수있으므로) epsilon 작업을 해주고, 무한대로 가면 price도 0으로합니다
epsilon = 1e-10
train_values = train_sample.iloc[:, 7:].values
train_values[train_values == 0] = epsilon
price = sales_sample.iloc[:, 7:].values / train_values
price[~np.isfinite(price)] = 0
# Create a new DataFrame for price

price_df = pd.DataFrame(price, columns=train_sample.columns[7:])
price_df.insert(0, 'ID', train_sample['ID'])

for col in price_df.select_dtypes(include=['float64']).columns:
    price_df[col] = price_df[col].astype('float32')

# Save the price DataFrame to a CSV file
price_path = './10.omega/temp_output/sales_price.csv'
price_df.to_csv(price_path, index=False)

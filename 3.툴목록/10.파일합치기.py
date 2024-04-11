# 1. 결과물 불러오기
type_1 = pd.read_csv('/data/user/postreview/type_output/baseline_submit_type_1.csv')
type_2 = pd.read_csv('/data/user/postreview/type_output/baseline_submit_type_2.csv')
type_3 = pd.read_csv('/data/user/postreview/type_output/baseline_submit_type_3.csv')

# 2. 겹치는지 확인하고 겹치면 대치하기
type_1_dict = {}
for index, row in tqdm(type_1.iterrows()):
    typeid1_name = row[0]
    typeid1_data = row[1:].values
    type_1_dict[typeid1_name] = typeid1_data

type_2_dict = {}
for index, row in tqdm(type_2.iterrows()):
    typeid2_name = row[0]
    typeid2_data = row[1:].values
    type_2_dict[typeid2_name] = typeid2_data

for index, row in tqdm(type_3.iterrows()):
    id_name = row[0]  # 브랜드 column
    if id_name in type_1_dict:
        type_3.iloc[index, 1:] = type_1_dict[id_name]
    elif id_name in type_2_dict:
        type_3.iloc[index, 1:] = type_2_dict[id_name]

finaloutput_path = '/data/user/postreview/output/baseline_submit_final_informerlstmcnn.csv'
type_3.to_csv(finaloutput_path, index=False)

## 최종 결과물
finaloutput_path
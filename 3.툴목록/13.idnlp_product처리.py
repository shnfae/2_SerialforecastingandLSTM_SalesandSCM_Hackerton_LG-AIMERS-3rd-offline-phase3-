import pandas as pd
from tqdm import tqdm

# Sample Data
df = pd.read_csv("./codereview/id_info_test.csv")
id_info = []
oneblank_list = []
count_duplicates = 0
double_list = []
double_id_list = [] 
matched_list = []
unmatched_list = []
dict_list = ["종류", "형태", "사용횟수","용도","용기","용량","연령","타입","타겟","수량","기능","매수","구성","제품용량","제품유형","타겟연령","재질","섭취방법","제품타입","섭취횟수","헤어타입","포장형태","매수","정수과정","두피타입","특징","겹수","롤수","종류","중량"]

grouped_dict = {
    "human": ["성인남녀", "신생아","임산부"],
    "종류": ["곡물", "액상","구미/젤리"],
    "원료명(식약처고시)": ["아르기닌","아연","프로틴","판토텐산","카제인","60억","20억","2억","비타민B6","BCAA"],
    "인증": ["이력추적관리"],
    "종류": ["분말"],
    "섭취방법": ["물에섞어서","바로음용"],
    "섭취횟수": ["하루두번","하루한번"],
    "효과":["기억력","혈행개선"]
}

info_dict = {value: key for key, values in grouped_dict.items() for value in values}



# Iterate through each row in the dataframe
for _, row in tqdm(df.iterrows()):
    info_str = row['info']
    
    # If info column is empty
    if pd.isna(info_str):
        id_info.append([])
        continue

    # Split the string based on ':'
    pairs = info_str.split(':')

    idx_info = []

    for i in range(len(pairs) - 1):
        # Split based on '_' to get index
        index = pairs[i].split('_')[-1]

        if len(index) > 10:
            if index[0:4] == "원료명(":
                index="원료명(식약처고시)"
            if index[0:4] == "기능성(":
                index="기능성(식약처고시)"
            else :
                index = index[:4]
        if index[:2] == index[2:4]:
            index = index[:1]
        if len(index) > 5 and index[:2] in dict_list:
            index = index[:2]
        if len(index) == 4 and index[:2] in dict_list and index[2:4] in dict_list:
            index = index[:2]
        if len(index) == 6 and index[:2] in dict_list and index[2:4] in dict_list:
            index = index[:2]
        if len(index) == 8 and index[:4] in dict_list :
            index = index[:4]
        if len(index) == 8 and index[:2] in dict_list :
            index = index[:2]
        if index[0:2] == "연령" or index[0:2] == "용량":
            index = index[:2]
        if index[0:3] == "섭취량" or index[0:3] == "사이즈" or index[0:3] == "향계열" :
            index = index[:3]
        if index[0:6] == "주요제품특징" or index[0:6] == "세부제품특징" or index[0:6]=="제품타입사이즈" or index[0:6]=="제품타입사이즈" :
            index = index[:6]
        if index[0:4] == "최소연령" or index[0:4] == "사용횟수" or index[0:4] == "타겟연령" or index[0:4] =="타켓연령" or index[0:4] == "보관방법" or index[0:4] == "사용부위" or index[0:4] == "사용대상" or index[0:4] == "헤어타입" or index[0:4] == "살균방식" :
            index = index[:4]
        if index[0:2] == "연령" or index[0:4] == "연령최소" or index[0:4] == "최소연령" or index[0:4] == "타켓연령" or index[0:4] == "사용대상"  or index[0:4] == "성별연령" or index[0:6] == "권장섭취연령" :
            index = "human"
        if index[0:4] == "제품형태" or index[0:4] == "포장형태" or index[0:2] == "용기":
            index = "형태"
        if index[0:4] == "제품타입":
            index = "제품타입"
        if index[0:6] == "주요제품특징" or index[0:6] == "세부제품특징":
            index = "특징"
        if index[0:4] == "날수부가":
            index = "날수"
        if index[0:4] == "칫솔종류":
            index = "칫솔종류"
        if index[0:1] == "품" :
            index = "품목"
        if index[0:1] == "타" :
            index = "타입"
        if index[0:1] == "종" or index[0:4] == "제품종류" or index[0:2] == "타입" or index[0:2] == "분류" or index[0:4] == "제품타입" or index[0:4] == "제품유형" :
            index = "종류"
        if index[0:1] == "특" :
            index = "특징"
        if index[0:1] == "구" :
            index = "구성"
        if index[0:1] == "기" or index =="사용효과" :
            index = "효과"
        if index[0:1] == "향" :
            index = "향계열"
        if index[0:1] == "연" :
            index = "human"

        # Split the next part of the pair to get info
        info = pairs[i + 1].split('_')[0]
        if index.startswith("A"):
            continue
        if info in info_dict:
            index = info_dict[info]

        if index and not info:
            # Check if this is the last pair by looking for another ':' after the current pair
            if i == len(pairs) - 2 or not pairs[i + 2]:
                continue
            else:
                oneblank_list.append([index, info])

        # Check if info is present but index is empty, and info starts with a number
        elif info and not index:
            if info[0].isdigit():
                index = "unit"
            else:
                if [index, info] not in oneblank_list:
                    oneblank_list.append([index, info])
                continue
        if info[-2:] == "월분" or info[-2:] == "일분":
            index = "제품용량"
        if info == "물과함께" :
            index = "섭취방법"
        if info == "면역력" or info == "눈건강" or info =="관절/뼈건강" or info=="전립선" or info=="영양보충" :
            index = "효과"
        if info[-1:] == "억" :
            index = "생균"


        if index == "human" :
            if  info[-2:] == "개월" or info == "베이비" or info == "신생아" or info == "1세"  :
                info = "신생아/영아"
            if  info[-1:] == "세" or info == "어린이용" or info[-2:] == "부터" or info == "어린이" or info[-2:] == "이상" or info[-3:] == "여아용" :
                info = "유아/청소년"
            if  info[-4:] == "남녀선택" or info[-2:] == "성인" :
                info = "성인남녀"
            if  info[-4:] == "성인여성" or info[-3:] == "임산부":
                info = "여성용"
            if  info[-4:] == "성인남성" :
                info = "남성용"
            if  info == "유아/청소년" or info == "신생아/영아" or info == "기타" or info =="1신생아/영아종류" or info=="신생아/영아human" or info=="신생아/영아종류":
                info = "미성년자"
        if info[:2] == "유아" or info[:2] == "아기" or info[-2:] == "젖병" or info[:2] == "분유" or info[:3] == "어린이"  :
            idx_info.append(["human", "미성년자"])
        if info[:2] == "요실" or info[:2] == "새치" or info[:2] == "탈모" or info[:3]=="갱년기"or info[:2]=="주름":
            idx_info.append(["human", "남녀공용"])
        if index == "1신생아/영아종류" or index=="신생아/영아종류" :
            index == "종류"

        # Append to oneblank_list if needed
        elif (index and not info) or (info and not index):
            if [index, info] not in oneblank_list:
                oneblank_list.append([index, info])
                continue

        idx_info.append([index, info])
    
    indices = [item[0] for item in idx_info]
    infos = [item[1] for item in idx_info]

    # Check for duplicate indices
    if len(indices) != len(set(indices)):
        double_list.append(idx_info)
        double_id_list.append(row.name)
        count_duplicates += 1

        indices = [item[0] for item in idx_info]
        infos = [item[1] for item in idx_info]
        if len(indices) != len(set(indices)):
            double_list.append(idx_info.copy())  # Save a copy of the original list with duplicates
            double_id_list.append(row.name)
            count_duplicates += 1

            unique_indices = []
            items_to_keep = []
            for idx, info in zip(indices, infos):
                if indices.count(idx) > 1 and idx not in unique_indices:
                    # This is a duplicate index. Retain only the first occurrence.
                    items_to_keep.append((idx, info))
                    unique_indices.append(idx)
                elif idx not in unique_indices:
                    # This is a unique index. Simply append it.
                    items_to_keep.append((idx, info))
                    unique_indices.append(idx)

            # Update idx_info with items to keep
            idx_info = items_to_keep

    id_info.append(idx_info)

print(len(id_info))
print(len(id_info))
print(len(oneblank_list))
print(len(double_list))
print(len(double_id_list))
print(len(matched_list))
print(len(unmatched_list))

print(len(id_info))


crosschange_dict = {
    "원료명(식약처고시)": ["아연", "마그네슘","철분","비타민D","비타민E","비타민C","비타민A","비오틴","철","마그네슘","칼슘","비타민B6","나이아신","비타민B2","판토텐산","셀레늄(셀렌)","카테킨","로르산","플라보노이드","식이섬유","지아잔틴","펩타이드","진세노사이드","루테인","실리마린"],
    "인증": ["식품품질"]
}


crosschange_dict2 = {
    "원료명(식약처고시)": ["DHA+EPA","단백질","엽산","히알루론산","코엔자임Q10","프락토올리고당","푸닉산+후코잔틴","감마리놀렌산","비타민B1","비타민B2","글루코사민","지방족알코올","대두액함량","비타민B6","공액리놀레산","키토산","인지질함량","엽록소","조단백","조지방","보스웰릭산","홍삼농축액","비타민C"],
}


specific_info_for_무첨가 = set()

for sublist in id_info:
    for idx, info in sublist:
        if idx == "무첨가":
            specific_info_for_무첨가.add(info)

# Change the index for specific info
for sublist in id_info:
    for i, (idx, info) in enumerate(sublist):
        if info in specific_info_for_무첨가:
            sublist[i] = ["무첨가", info]


for sublist in id_info:
    for i, (idx, info) in enumerate(sublist):
        if info == "저자극":
            sublist[i] = ["특징","저자극"]
        if idx == "비건인증":
            sublist[i] = ["인증", "비건인증"]
        if idx == "홍삼농축액2":
            sublist[i] = ["원료명(식약처고시)", "홍삼농축액"]
        if idx == "신생아/영아종류":
            sublist[i] = ["종류", info]
        if idx == "소재":
            sublist[i] = ["재질", info]
        if idx == "신생아/":
            if info == "3~6개월" or info == "신생아/영아" or info == "3~5세":
                sublist[i] = ["human", "미성년자"]
            else :
                sublist[i] = ["종류", info]
        for key, values in crosschange_dict.items():
            if idx in values:
                sublist[i] = [key, idx]
        for key, values in crosschange_dict2.items():
            if idx in values:
                sublist[i] = [key, idx]

double_list_after_transformation = []
matched_list_after_transformation = []
unmatched_list_after_transformation = []


indices = [item[0] for item in idx_info]
infos = [item[1] for item in idx_info]

# Check for duplicate indices
if len(indices) != len(set(indices)):
    double_list.append(idx_info)
    double_id_list.append(row.name)
    count_duplicates += 1

    indices = [item[0] for item in idx_info]
    infos = [item[1] for item in idx_info]
    if len(indices) != len(set(indices)):
        double_list.append(idx_info.copy())  # Save a copy of the original list with duplicates
        double_id_list.append(row.name)
        count_duplicates += 1

        unique_indices = []
        items_to_keep = []
        for idx, info in zip(indices, infos):
            if indices.count(idx) > 1 and idx not in unique_indices:
                # This is a duplicate index. Retain only the first occurrence.
                items_to_keep.append((idx, info))
                unique_indices.append(idx)
            elif idx not in unique_indices:
                # This is a unique index. Simply append it.
                items_to_keep.append((idx, info))
                unique_indices.append(idx)

        # Update idx_info with items to keep
        idx_info = items_to_keep

    id_info.append(idx_info)

all_indices = [item[0] for sublist in id_info for item in sublist]
all_infos = [item[1] for sublist in id_info for item in sublist]

# A dictionary to count occurrences of info for each index
index_info_count = {}

for idx, info in zip(all_indices, all_infos):
    if idx in index_info_count:
        index_info_count[idx].add(info)
    else:
        index_info_count[idx] = {info}

# Find the indices whose info count is 1
indices_to_remove = [idx for idx, infos in index_info_count.items() if len(infos) == 1]

# Filter out the id_info list
filtered_id_info = []
for sublist in id_info:
    new_sublist = [pair for pair in sublist if pair[0] not in indices_to_remove]
    filtered_id_info.append(new_sublist)

id_info = filtered_id_info



# Create a dictionary to store the counts of each info for each index
index_info_count = {}

# Iterate over all idx_info lists
for idx_list in tqdm(id_info):
    # Iterate over each [index, info] pair
    for idx, info in idx_list:
        # If index is not in the dictionary, initialize it
        if idx not in index_info_count:
            index_info_count[idx] = {}
        
        # If info is not in the sub-dictionary, initialize it
        if info not in index_info_count[idx]:
            index_info_count[idx][info] = 0
        
        # Increment the count
        index_info_count[idx][info] += 1

# Convert the dictionary into a list of lists
index_info_list = [[idx, [[info, count] for info, count in info_dict.items()]] for idx, info_dict in index_info_count.items()]

# Sort the info inside each index based on count
for _, info_list in index_info_list:
    info_list.sort(key=lambda x: x[1], reverse=True)

# Sort the indices based on the total sum of counts for their associated info
index_info_list.sort(key=lambda x: sum([info[1] for info in x[1]]), reverse=True)

print(len(index_info_list))
# Print the first 20 items of index_info_list
with open('index_info_list_14.txt', 'w', encoding='utf-8') as file:
    for i, (index_name, info_list) in enumerate(index_info_list):
        file.write(f"{i + 1}. Index Name: {index_name}\n")
        file.write("Associated Info:\n")
        for info, count in info_list:
            file.write(f"  {info}: {count}\n")
        file.write("-" * 50 + "\n")  # add a separator for clarity

print(len(id_info))

print(len(index_info_list))

print(id_info[9255])


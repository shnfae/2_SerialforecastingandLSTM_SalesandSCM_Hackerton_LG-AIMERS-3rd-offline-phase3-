import pandas as pd
import numpy as np

train_data = pd.read_csv('./train_type_2.csv').drop(columns=['제품'])


def compute_psfa(pred, actual, indexs_bigcat):
    psfa_list = []
    PSFA = 1
    max_index = actual.shape[0] - 1  # Maximum allowable index
    
    for cat in indexs_bigcat.keys():
        ids = indexs_bigcat[cat]
        ids = [idx for idx in ids if idx <= max_index]
        
        for day in range(3):
            total_sell = np.sum(actual[ids, day]) # day별 총 판매량
            pred_values = pred[ids, day] # day별 예측 판매량
            target_values = actual[ids, day] # day별 실제 판매량

            # 실제 판매와 예측 판매가 같은 경우 오차가 없는 것으로 간주
            denominator = np.maximum(target_values, pred_values)
            numerators = np.abs(target_values - pred_values)
            
            diffs = np.where(denominator!=0, numerators / denominator, 0)
            diffs[np.isnan(diffs)] = 0  # replace any NaN values with 0
            
            if total_sell != 0:
                sell_weights = target_values / total_sell  # Item별 day 총 판매량 내 비중
            else:
                sell_weights = np.ones_like(target_values) / len(ids)  # 1 / len(ids)로 대체

            if not np.isnan(diffs).any():  # diffs에 NaN이 없는 경우에만 PSFA 값 업데이트
                PSFA -= np.sum(diffs * sell_weights) / (3 * 5)
        
        psfa_list.append(PSFA)
        print(psfa_list)
        
    psfa_mean = psfa_list[4]
    return psfa_mean

def calculate_psfa(submission_path, actual_data_path, indexs_bigcat):
    submission_df = pd.read_csv(submission_path).drop(columns=['ID'])
    actual_df = pd.read_csv(actual_data_path).drop(columns=['ID'])

    # Assuming the sales predictions are in the columns of submission_df
    # and the actual sales are in the columns of actual_df
    pred_sales = submission_df.values
    actual_sales = actual_df.values

    psfa_score = compute_psfa(pred_sales, actual_sales, indexs_bigcat)

    return psfa_score



indexs_bigcat={}
for bigcat in train_data['대분류'].unique():
    indexs_bigcat[bigcat] = list(train_data.loc[train_data['대분류']==bigcat].index)



score = calculate_psfa('true_baseline_submit_type_2_0_223_120,10.csv', './actual_3_2.csv', indexs_bigcat)
print(score)


indexs_bigcat.keys()
print(indexs_bigcat.keys())


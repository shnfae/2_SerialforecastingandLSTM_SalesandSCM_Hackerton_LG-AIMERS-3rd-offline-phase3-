import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.mx import DeepAREstimator, Trainer
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.mx.trainer.learning_rate_scheduler import LearningRateReduction
from gluonts.mx.trainer.model_averaging import ModelAveraging, SelectNBestSoftmax, SelectNBestMean
from gluonts.mx.distribution import StudentTOutput, GaussianOutput, LaplaceOutput, GenParetoOutput
from tqdm import tqdm
import datetime
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder

CFG = {
    'TRAIN_WINDOW_SIZE':200,
    'PREDICT_SIZE':21,
    'EPOCHS':10,
    'LEARNING_RATE':1e-4,
    'BATCH_SIZE':512,
    'SEED':41
}


train_data = pd.read_csv('/data/user/postreview/temp_output/train_1.csv').drop(columns=['ID','중분류', '제품'],errors='coerce')

print("success")

scale_max_dict = {}
scale_min_dict = {}
for idx in tqdm(range(len(train_data))):
    maxi = np.max(train_data.iloc[idx,4:])
    mini = np.min(train_data.iloc[idx,4:])
    if maxi == mini :
        train_data.iloc[idx,4:] = 0
    else:
        train_data.iloc[idx,4:] = (train_data.iloc[idx,4:] - mini) / (maxi - mini)
    scale_max_dict[idx] = maxi
    scale_min_dict[idx] = mini

print("success")


label_encoder = LabelEncoder()
categorical_columns = ['대분류', '소분류', '브랜드']

for col in categorical_columns:
    label_encoder.fit(train_data[col])
    train_data[col] = label_encoder.transform(train_data[col])




def deepar_train_data(data, train_size=CFG['TRAIN_WINDOW_SIZE'], predict_size=CFG['PREDICT_SIZE'], feature_size=3):
    num_rows = len(data)
    window_size = train_size + predict_size
    final_data = []
    
    for i in tqdm(range(num_rows)):
        sales_data = np.array(data.iloc[i, feature_size:])
        date_data = datetime.datetime(2022, 1, 1)
        bigcat_data = int(data.iloc[i, 0])
        smacat_data = int(data.iloc[i, 1])
        brand_data = int(data.iloc[i, 2])

        
        for j in range(len(sales_data) - window_size + 1):
            window = sales_data[j: j + window_size]
            current_date = date_data + datetime.timedelta(days=j)
            
            # Create a dictionary for each windowed data point
            data_dict = {
                "start": current_date.strftime('%Y-%m-%d %H:%M'),
                "target": window[:train_size].tolist(),
                "cat" : [i,bigcat_data,smacat_data,brand_data]  # Convert numpy array to list
            }
            
            final_data.append(data_dict)
    
    return ListDataset(final_data, freq="1D")


def deepar_predict_data(data, train_size=CFG['TRAIN_WINDOW_SIZE'], predict_size=CFG['PREDICT_SIZE'], feature_size=4):
    num_rows = len(data)
    predict_data = []
    
    for i in tqdm(range(num_rows)):
        sales_data = np.array(data.iloc[i, -train_size:])
        window_sales = sales_data[-train_size : ]
        date_data = datetime.datetime(2023, 4, 5)
        current_date = date_data - datetime.timedelta(days=train_size)

        bigcat_data = int(data.iloc[i, 0])
        smacat_data = int(data.iloc[i, 1])
        brand_data = int(data.iloc[i, 2])

        data_dict = {
            "start": current_date.strftime('%Y-%m-%d %H:%M'),
            "target": window_sales[-train_size : ].tolist(),
            "cat" : [i,bigcat_data,smacat_data,brand_data]  # Convert numpy array to list
        }
        predict_data.append(data_dict)
    
    return ListDataset(predict_data, freq="1D")

train_input = deepar_train_data(train_data)
test_input = deepar_predict_data(train_data)


print(train_input[255])

print("success")

callbacks = [
    ModelAveraging(avg_strategy=SelectNBestMean(num_models=10))
]

estimator = DeepAREstimator(
    freq="1D", #
    prediction_length=CFG['PREDICT_SIZE'],
    context_length=CFG['TRAIN_WINDOW_SIZE'],
    learning_rate=CFG['LEARNING_RATE'],
    num_cells=40,
    num_layers=2,
    dropout_rate=0.1,
    scaling= False,
    distr_output=GaussianOutput(),
    trainer=Trainer(epochs=CFG['EPOCHS'],
                    callbacks=callbacks)
)

model = estimator.train(train_input)
predictor = estimator.train(test_input)

num_samples = 100
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_input,  # using your training data as an example, replace with test data if needed
    predictor=predictor,
    num_samples=num_samples
)

forecasts = list(forecast_it)
tss = list(ts_it)

# Extract mean predictions from the samples
mean_forecasts = [f.mean for f in forecasts]

# Convert predictions to DataFrame
df_predictions = pd.DataFrame(mean_forecasts)

for idx, row in df_predictions.iterrows():
    df_predictions.loc[idx] = row * (scale_max_dict[idx] - scale_min_dict[idx]) + scale_min_dict[idx]
    
pred = df_predictions.round(0).astype(int)

pred.to_csv('/data/user/codereview/type_output/baseline_submit_deepar2.csv', index=False)


# Save to CSV
submit = pd.read_csv('/data/user/codereview/temp_output/sample_submission_1.csv')
submit.head()
submit.iloc[:,1:] = pred
submit.head()
submit.to_csv('/data/user/codereview/type_output/baseline_submit_deepar2sample.csv', index=False)

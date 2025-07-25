import numpy as np
import pandas as pd

file_path1 = "C:/Users/LENOVO/Downloads/DATA FLOWABILITY REVOLUTION coba.csv"
file_path2 = "C:/Users/LENOVO/Downloads/Data Param (1).csv"
file_path3 = "C:/Users/LENOVO/Downloads/Data Filling A2 22 Juni - 14 Juli.csv"

dataflow = pd.read_csv(file_path1)
dataparam1 = pd.read_csv(file_path2)
dataparam2 = pd.read_csv(file_path3)

dataparam = pd.concat([dataparam1, dataparam2], axis=0)
dataparam = dataparam.drop_duplicates(subset='dtmInsertedDate')

dataflow = dataflow[['NO BO', 'DESC', 'SL Ratio','Avalanche Energy mJ/kg','Avalanche Angle (deg)','Dynamic Density g/cc','Bulk Density (weight/TD)']]

dataparam = dataparam[dataparam["intMachineSpeed"]>10]

def calculate_weight_duration_per_source(dataparam):

    dataparam['dtmInsertedDate'] = pd.to_datetime(dataparam['dtmInsertedDate'], errors='coerce')

    df = dataparam[['txtSourceNumber', 'intWeight', 'dtmInsertedDate']].dropna().copy()
    df.sort_values(['txtSourceNumber', 'dtmInsertedDate'], inplace=True)

    df['next_time'] = df.groupby('txtSourceNumber')['dtmInsertedDate'].shift(-1)
    df['duration'] = (df['next_time'] - df['dtmInsertedDate']).dt.total_seconds().fillna(0)

    result = df.groupby(['txtSourceNumber', 'intWeight'])['duration'].sum().reset_index()

    result = result.sort_values(['txtSourceNumber', 'duration'], ascending=[True, False])

    return result

duration_df = calculate_weight_duration_per_source(dataparam)

results = []
for source in duration_df['txtSourceNumber'].unique():
    df_filtered = duration_df[duration_df['txtSourceNumber'] == source]
    longest_row = df_filtered.loc[df_filtered['duration'].idxmax()]
    param = longest_row['intWeight']
    duration = longest_row['duration']
    results.append({
        'txtSourceNumber': source,
        'intWeight': param,
        'duration_seconds': duration
    })

result_df = pd.DataFrame(results)


datajoin = pd.merge(dataflow, result_df, left_on='NO BO', right_on='txtSourceNumber', how='left')


import pandas as pd
import re

def extract_weight_info(desc):
    weights = re.findall(r'\d+', str(desc))
    weights = [int(w) for w in weights]

    if not weights:
        return 'UNGROUP', None

    weight = max(weights)

    if weight < 100:
        category = '0'
    elif 150 <= weight <= 200:
        category = 'A'
    elif 270 <= weight <= 299:
        category = 'B'
    elif weight >= 300:
        category = 'C'
    else:
        category = 'UNGROUP'

    return category, weight


datajoin[['WeightCategory', 'targetweight']] = datajoin['DESC'].apply(
    lambda x: pd.Series(extract_weight_info(x))
)


dataA = datajoin[datajoin['WeightCategory'] == 'A']
dataB = datajoin[datajoin['WeightCategory'] == 'B']
dataC = datajoin[datajoin['WeightCategory'] == 'C']
dataD = datajoin[datajoin['WeightCategory'] == 'D']

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
import xgboost as xgb

# Dataset
datates1 = dataC
datates = datates1[datates1['intWeight'].notnull()]

X_full = datates[[  
    'SL Ratio',  
    'Avalanche Energy mJ/kg',  
    'Avalanche Angle (deg)',  
    'Dynamic Density g/cc',  
    'Bulk Density (weight/TD)',  
    'targetweight'  
]]

y_full = datates['intWeight']

# Hyperparameters to try
test_sizes = [0.2, 0.3]
random_states = [10, 20, 30,42]

best_overall_mae = float('inf')
best_overall_model = None
best_overall_params = {}

# MAE scorer 
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# XGBoost param grid
# XGBoost param grid
param_grid = {
    'n_estimators': [25, 50, 100,150, 200,250, 300],
    'max_depth': [3, 5, 7,9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

from sklearn.metrics import r2_score

for test_size in test_sizes:
    for random_state in random_states:
        print(f"\n🔷 Trying test_size={test_size}, random_state={random_state}...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=test_size, random_state=random_state
        )

        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring=mae_scorer,
            cv=3,
            verbose=0,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        best_xgb = grid_search.best_estimator_
        best_params = grid_search.best_params_

        y_pred = best_xgb.predict(X_test)
        y_train_pred = best_xgb.predict(X_train)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_pred)

        print(f"✅ Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f}")
        print(f"✅ Train R²:  {train_r2:.4f} | Test R²:  {test_r2:.4f}")

        if test_mae < best_overall_mae:
            best_overall_mae = test_mae
            best_overall_model = best_xgb
            best_overall_params = {
                'test_size': test_size,
                'random_state': random_state,
                **best_params,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2
            }

print("\n🏆 Best Overall Parameters:")
print(best_overall_params)

import math
user_input = input("Masukkan nilai X (SL Ratio, AE, AA, DD, BD, TargetWeight): ")

# Ubah ke array dan reshape
X_coba = np.array([float(x.strip()) for x in user_input.split(',')]).reshape(1, -1)

# Prediksi
y_coba = best_overall_model.predict(X_coba)
y_coba_value = y_coba[0]

# Bulatkan ke atas ke kelipatan 10
y_coba_rounded = math.ceil(y_coba_value / 10) * 10

# Tampilkan hasil
print(f"Prediksi berat (dibulatkan ke atas ke kelipatan 10): {y_coba_rounded}")
print(f"Hasil prediksi asli: {y_coba_value:.2f}")